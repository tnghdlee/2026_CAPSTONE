// =============================================================
// Project : 생활도로(골목길)에서의 전방 추돌 예측 및 방지 알고리즘 (C++ 포팅)
// Original: main_code.py (Python + Ultralytics YOLOv8m)
// Team    : 김찬진, 양우영, 엄유빈, 이수홍
// Dataset : AI Hub "생활도로 객체인식 자율주행 데이터" (2023)
// Model   : YOLOv8m -> ONNX 변환 후 OpenCV DNN으로 추론
// Build   : CMakeLists.txt 참고 (C++17, OpenCV 4.8 이상 — 4.7 이하는 YOLOv8 ONNX를 읽지 못함)
// Run     : fcw <best.onnx> <image_dir> [--no-display]
//           --no-display : 화면 출력과 waitKey(30) 대기를 끄고 순수 처리 시간만 측정
// =============================================================

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ==========================================
// 1. 설정 (Python 원본과 동일한 값)
// ==========================================
constexpr float  CONF_THRESHOLD    = 0.5f;  // YOLO 객체 검출 신뢰도 임계값
constexpr float  NMS_IOU_THRESHOLD = 0.7f;  // Ultralytics 기본 NMS IoU
constexpr int    INPUT_SIZE        = 640;   // ONNX 입력 해상도
constexpr int    FPS               = 10;    // 프레임 레이트 (10 FPS)
constexpr size_t FRAME_INTERVAL    = 10;    // TTC/횡방향 분석 윈도우 (10프레임 ≈ 1초)

// 클래스 ID: 0=Pedestrian, 1=Vehicle, 2=PM
enum ObjClass { PEDESTRIAN = 0, VEHICLE = 1, PM = 2 };

struct Detection     { float x1, y1, x2, y2; float conf; int cls; };
struct TrackedObject { float x1, y1, x2, y2; int id; int cls; };
struct HistoryEntry  { float area; int cx; };

struct RiskResult {
    std::vector<cv::Point> ref_points;
    int total_risk = 0;
    std::string discount_text;  // Python 원본과 동일하게 계산만 하고 화면에는 표시하지 않음
};

// 프레임마다 한 번 계산되는 ROI 정보
struct FrameContext {
    std::vector<cv::Point> main_roi;
    std::vector<cv::Point> buffer_roi;
    int    cx_roi = 0, cy_roi = 0;
    double max_dist = 1.0;
    double ego_turn_threshold = 0.0;
    double lateral_threshold = 0.0;
};

// ==========================================
// 파일 입출력 (Windows 한글 경로 대응)
// cv::imread와 경로 기반 readNet은 Windows에서 한글 경로를 읽지 못하므로
// 바이트로 읽은 뒤 메모리에서 디코딩한다.
// ==========================================
static std::vector<uchar> readBytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize size = f.tellg();
    if (size <= 0) return {};
    std::vector<uchar> buf(static_cast<size_t>(size));
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(buf.data()), size)) return {};
    return buf;  // 파일 크기만큼 한 번에 읽기 (바이트 단위 반복 읽기보다 빠름)
}

static cv::Mat readImage(const fs::path& p) {
    std::vector<uchar> buf = readBytes(p);
    if (buf.empty()) return cv::Mat();
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// ==========================================
// YOLOv8 ONNX 검출기 (Ultralytics 전처리/후처리 재현)
// ==========================================
class YoloOnnxDetector {
public:
    explicit YoloOnnxDetector(const fs::path& onnx_path) {
        std::vector<uchar> buf = readBytes(onnx_path);
        if (buf.empty()) throw std::runtime_error("cannot read model file");
        net_ = cv::dnn::readNetFromONNX(buf);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<Detection> detect(const cv::Mat& frame) {
        float scale = 1.f;
        int pad_left = 0, pad_top = 0;
        cv::Mat input = letterbox(frame, scale, pad_left, pad_top);

        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE),
                                              cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        // YOLOv8 출력: [1, 4+nc, 8400] -> [8400, 4+nc]
        cv::Mat out = outs[0];
        CV_Assert(out.dims == 3);
        const int d1 = out.size[1], d2 = out.size[2];
        cv::Mat pred(d1, d2, CV_32F, out.ptr<float>());
        if (d1 < d2) pred = pred.t();
        const int num_classes = pred.cols - 4;

        std::vector<Detection> cands;
        std::vector<cv::Rect2d> nms_boxes;
        std::vector<float> scores;
        constexpr float MAX_WH = 7680.f;  // 클래스별 NMS를 위한 좌표 오프셋 (Ultralytics 방식)

        for (int i = 0; i < pred.rows; ++i) {
            const float* row = pred.ptr<float>(i);
            const float* cls_scores = row + 4;
            const int cls = static_cast<int>(std::max_element(cls_scores, cls_scores + num_classes) - cls_scores);
            const float score = cls_scores[cls];
            if (score <= CONF_THRESHOLD) continue;

            const float cx = row[0], cy = row[1], w = row[2], h = row[3];
            float x1 = (cx - w / 2.f - pad_left) / scale;
            float y1 = (cy - h / 2.f - pad_top) / scale;
            float x2 = (cx + w / 2.f - pad_left) / scale;
            float y2 = (cy + h / 2.f - pad_top) / scale;
            x1 = std::clamp(x1, 0.f, static_cast<float>(frame.cols));
            x2 = std::clamp(x2, 0.f, static_cast<float>(frame.cols));
            y1 = std::clamp(y1, 0.f, static_cast<float>(frame.rows));
            y2 = std::clamp(y2, 0.f, static_cast<float>(frame.rows));

            cands.push_back({x1, y1, x2, y2, score, cls});
            nms_boxes.emplace_back(x1 + cls * MAX_WH, y1, x2 - x1, y2 - y1);
            scores.push_back(score);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_boxes, scores, CONF_THRESHOLD, NMS_IOU_THRESHOLD, keep);

        std::vector<Detection> dets;
        dets.reserve(keep.size());
        for (int idx : keep) dets.push_back(cands[idx]);
        return dets;
    }

private:
    // Ultralytics LetterBox와 동일: 비율 유지 리사이즈 + 회색(114) 패딩
    static cv::Mat letterbox(const cv::Mat& img, float& scale, int& pad_left, int& pad_top) {
        scale = std::min(static_cast<float>(INPUT_SIZE) / img.rows, static_cast<float>(INPUT_SIZE) / img.cols);
        const int new_w = static_cast<int>(std::round(img.cols * scale));
        const int new_h = static_cast<int>(std::round(img.rows * scale));
        const float dw = (INPUT_SIZE - new_w) / 2.f;
        const float dh = (INPUT_SIZE - new_h) / 2.f;

        cv::Mat resized;
        if (new_w != img.cols || new_h != img.rows)
            cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        else
            resized = img;

        const int top    = static_cast<int>(std::round(dh - 0.1f));
        const int bottom = static_cast<int>(std::round(dh + 0.1f));
        const int left   = static_cast<int>(std::round(dw - 0.1f));
        const int right  = static_cast<int>(std::round(dw + 0.1f));
        cv::Mat out;
        cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        pad_left = left;
        pad_top = top;
        return out;
    }

    cv::dnn::Net net_;
};

// ==========================================
// IoU 기반 객체 추적기
// Ultralytics model.track(persist=True)의 ID 유지 역할을 대체한다.
// - IoU가 높은 순서로 탐욕적 매칭
// - 새 트랙은 다음 프레임에서 한 번 더 매칭되어야 확정(첫 프레임은 즉시 확정)
// - 확정 트랙은 최대 30프레임까지 놓쳐도 ID 유지
// ==========================================
class IouTracker {
public:
    std::vector<TrackedObject> update(const std::vector<Detection>& dets) {
        ++frame_id_;
        std::vector<std::tuple<float, size_t, size_t>> pairs;  // (IoU, track idx, det idx)
        for (size_t t = 0; t < tracks_.size(); ++t)
            for (size_t d = 0; d < dets.size(); ++d) {
                const float v = iou(tracks_[t].box, toRect(dets[d]));
                if (v > MATCH_IOU) pairs.emplace_back(v, t, d);
            }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

        std::vector<bool> t_used(tracks_.size(), false), d_used(dets.size(), false);
        std::vector<int> det_to_track(dets.size(), -1);
        for (const auto& [v, t, d] : pairs) {
            if (t_used[t] || d_used[d]) continue;
            t_used[t] = d_used[d] = true;
            Track& tr = tracks_[t];
            tr.box = toRect(dets[d]);
            tr.cls = dets[d].cls;
            tr.lost = 0;
            tr.confirmed = true;
            det_to_track[d] = tr.id;
        }

        // 매칭되지 않은 트랙: 미확정이면 삭제, 확정이면 lost 증가
        std::vector<Track> kept;
        for (size_t t = 0; t < tracks_.size(); ++t) {
            Track tr = tracks_[t];
            if (!t_used[t]) {
                if (!tr.confirmed) continue;
                if (++tr.lost > MAX_LOST) continue;
            }
            kept.push_back(tr);
        }
        tracks_ = std::move(kept);

        // 매칭되지 않은 검출: 새 트랙 생성
        for (size_t d = 0; d < dets.size(); ++d) {
            if (d_used[d]) continue;
            const bool confirmed_now = (frame_id_ == 1);
            tracks_.push_back({next_id_, toRect(dets[d]), dets[d].cls, 0, confirmed_now});
            if (confirmed_now) det_to_track[d] = next_id_;
            ++next_id_;
        }

        // 이번 프레임에서 확정 트랙과 연결된 검출만 출력 (검출 순서 유지)
        std::vector<TrackedObject> out;
        for (size_t d = 0; d < dets.size(); ++d) {
            if (det_to_track[d] < 0) continue;
            out.push_back({dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2, det_to_track[d], dets[d].cls});
        }
        return out;
    }

private:
    struct Track { int id; cv::Rect2f box; int cls; int lost; bool confirmed; };
    static constexpr float MATCH_IOU = 0.2f;
    static constexpr int   MAX_LOST  = 30;

    static cv::Rect2f toRect(const Detection& d) { return {d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1}; }
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        const float inter = (a & b).area();
        const float uni = a.area() + b.area() - inter;
        return uni > 0.f ? inter / uni : 0.f;
    }

    std::vector<Track> tracks_;
    int next_id_ = 1;
    long long frame_id_ = 0;
};

// ==========================================
// 2-1. 주행 경로(ROI) 설정 및 중심점 도출
// ==========================================
static FrameContext buildFrameContext(int width, int height) {
    FrameContext ctx;
    ctx.ego_turn_threshold = width * 0.1;    // 자차 회전 판정 임계값 (화면 폭의 10%)
    ctx.lateral_threshold  = width * 0.015;  // 측면 객체 이동 판정 임계값 (화면 폭의 1.5%)
    ctx.main_roi = {{static_cast<int>(width * 0.3), height},
                    {static_cast<int>(width * 0.45), static_cast<int>(height * 0.65)},
                    {static_cast<int>(width * 0.55), static_cast<int>(height * 0.65)},
                    {static_cast<int>(width * 0.7), height}};
    ctx.buffer_roi = {{static_cast<int>(width * 0.2), height},
                      {static_cast<int>(width * 0.40), static_cast<int>(height * 0.65)},
                      {static_cast<int>(width * 0.60), static_cast<int>(height * 0.65)},
                      {static_cast<int>(width * 0.8), height}};

    const cv::Moments M = cv::moments(ctx.main_roi);
    if (M.m00 != 0) {
        ctx.cx_roi = static_cast<int>(M.m10 / M.m00);
        ctx.cy_roi = static_cast<int>(M.m01 / M.m00);
    } else {
        ctx.cx_roi = static_cast<int>(width * 0.5);
        ctx.cy_roi = static_cast<int>(height * 0.8);
    }
    ctx.max_dist = 0.0;
    for (const auto& pt : ctx.main_roi)
        ctx.max_dist = std::max(ctx.max_dist, std::hypot(pt.x - ctx.cx_roi, pt.y - ctx.cy_roi));
    return ctx;
}

// ==========================================
// 2-3 ~ 2-6. 판정점 설정, TTC 계산, 정량적 위험도 산출 (Python 원본 로직 그대로)
// ==========================================
static bool insidePolygon(const std::vector<cv::Point>& poly, const cv::Point& p) {
    return cv::pointPolygonTest(poly, cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)), false) >= 0;
}

static RiskResult computeRisk(const TrackedObject& o, const FrameContext& ctx,
                              std::unordered_map<int, std::deque<HistoryEntry>>& track_history) {
    RiskResult r;
    const float x1 = o.x1, y1 = o.y1, x2 = o.x2, y2 = o.y2;
    const float w = x2 - x1;
    const float h = y2 - y1;
    const float area = w * h;
    const int current_cx = static_cast<int>(x1 + w / 2);

    // 2-3. 판정점(ref_points) 설정 및 ROI 진입 판정
    if (o.cls == VEHICLE) {
        r.ref_points = {{static_cast<int>(x1 + w / 2), static_cast<int>(y2)},
                        {static_cast<int>(x1), static_cast<int>(y1 + h / 2)},
                        {static_cast<int>(x2), static_cast<int>(y1 + h / 2)}};
    } else {
        r.ref_points = {{static_cast<int>(x1), static_cast<int>(y2)},
                        {static_cast<int>(x2), static_cast<int>(y2)}};
    }

    bool is_in_main = false, is_in_buffer = false;
    for (const auto& p : r.ref_points) {
        is_in_main   = is_in_main   || insidePolygon(ctx.main_roi, p);
        is_in_buffer = is_in_buffer || insidePolygon(ctx.buffer_roi, p);
    }

    // 2-4. 데이터 누적 및 윈도우 기반 분석
    double ttc_sec = std::numeric_limits<double>::infinity();
    int lateral_movement = 0;
    bool is_moving_away = false;

    auto& hist = track_history[o.id];
    hist.push_back({area, current_cx});
    if (hist.size() > FRAME_INTERVAL) hist.pop_front();

    if (hist.size() == FRAME_INTERVAL) {
        const HistoryEntry& prev = hist.front();
        if (prev.area > 0) {
            const double total_expansion_rate = (area - prev.area) / prev.area;
            if (total_expansion_rate > 0) {
                const double rate_per_sec = (total_expansion_rate / FRAME_INTERVAL) * FPS;
                ttc_sec = 2.0 / rate_per_sec;
            }
        }
        lateral_movement = std::abs(current_cx - prev.cx);
        // ROI 무게중심으로부터 가로 방향으로 멀어지는지 판별
        if (std::abs(current_cx - ctx.cx_roi) > std::abs(prev.cx - ctx.cx_roi)) is_moving_away = true;
    }

    // 2-5. 정량적 위험도 계산
    // (1) 공간적 점수 (최대 40점): R_spatial = 20 + 20 * (1 - d/d_max)
    int spatial_score = 0;
    if (is_in_main) {
        double min_dist = std::numeric_limits<double>::infinity();
        bool has_valid = false;
        for (const auto& p : r.ref_points) {
            if (!insidePolygon(ctx.main_roi, p)) continue;
            has_valid = true;
            min_dist = std::min(min_dist, std::hypot(p.x - ctx.cx_roi, p.y - ctx.cy_roi));
        }
        if (has_valid) {
            const double dist_ratio = std::max(0.0, std::min(1.0 - (min_dist / ctx.max_dist), 1.0));
            spatial_score = static_cast<int>(20 + (20 * dist_ratio));
        } else {
            spatial_score = 20;
        }
    } else if (is_in_buffer) {
        spatial_score = 20;
    }

    // (2) 시간적 점수 (최대 50점): TTC 역수 기반 비선형 정규화
    int temporal_score = 0;
    double min_ttc, max_ttc;
    if (o.cls == VEHICLE) { min_ttc = 1.5; max_ttc = 4.0; }  // 차량 : FCW 표준
    else                  { min_ttc = 2.5; max_ttc = 5.0; }  // 보행자/PM : Euro NCAP 기준

    if (ttc_sec <= min_ttc)      temporal_score = 50;
    else if (ttc_sec >= max_ttc) temporal_score = 0;
    else temporal_score = static_cast<int>(50 * ((1 / ttc_sec - 1 / max_ttc) / (1 / min_ttc - 1 / max_ttc)));

    if (o.cls == VEHICLE && !is_in_main) temporal_score = 0;

    // (3) 객체 가중치 (최대 10점): 교통약자 보호
    const int object_weight = (o.cls == PEDESTRIAN || o.cls == PM) ? 10 : 0;

    // (4) 최종 위험도 (0~100점)
    int total_risk = std::min(100, spatial_score + temporal_score + object_weight);

    // 2-6. 횡방향 이탈 세분화
    if (lateral_movement > ctx.ego_turn_threshold) {
        total_risk = static_cast<int>(total_risk * 0.2);
        r.discount_text = " (-Turn Ignore)";
    } else if (lateral_movement > ctx.lateral_threshold) {
        if (is_moving_away) {
            total_risk = static_cast<int>(total_risk * 0.6);
            r.discount_text = " (-Pass By)";
        } else {
            r.discount_text = " (!CUT-IN!)";
        }
    }

    r.total_risk = total_risk;
    return r;
}

// ==========================================
// 2-7. 상태 판정 및 화면 시각화 (UI)
// ==========================================
static void drawObject(cv::Mat& frame, const TrackedObject& o, const RiskResult& r,
                       std::vector<cv::Rect>& drawn_label_rects) {
    const int width = frame.cols, height = frame.rows;

    for (const auto& p : r.ref_points) cv::circle(frame, p, 5, cv::Scalar(255, 255, 255), -1);

    std::string status;
    cv::Scalar color;
    int thickness;
    if (r.total_risk >= 80)     { status = "DANGER";  color = cv::Scalar(0, 0, 255);   thickness = 4; }  // 빨강
    else if (r.total_risk > 40) { status = "CAUTION"; color = cv::Scalar(0, 165, 255); thickness = 2; }  // 주황
    else                        { status = "SAFE";    color = cv::Scalar(255, 0, 0);   thickness = 2; }  // 파랑

    // 바운딩 박스
    cv::rectangle(frame, cv::Point(static_cast<int>(o.x1), static_cast<int>(o.y1)),
                  cv::Point(static_cast<int>(o.x2), static_cast<int>(o.y2)), color, thickness);

    const char* obj_name = o.cls == PEDESTRIAN ? "Pedestrian" : o.cls == VEHICLE ? "Vehicle" : o.cls == PM ? "PM" : "Unknown";
    const std::string display_text = std::string(obj_name) + "|Risk:" + std::to_string(r.total_risk) + "%";

    // 다수 객체가 밀집한 골목길 환경에서 라벨이 겹치지 않도록
    // 박스 안쪽 좌상단에서 시작해 계단식으로 아래로 이동시키는 배치 로직
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 1.0;
    const int font_thickness = 2;
    const cv::Scalar text_color(255, 255, 255);

    int baseline = 0;
    const cv::Size ts = cv::getTextSize(display_text, font, font_scale, font_thickness, &baseline);
    const int text_w = ts.width, text_h = ts.height;

    const int pad_x = 4, pad_y = 3;
    const int label_box_w = text_w + pad_x * 2;
    const int label_box_h = text_h + pad_y * 2;

    int label_x = static_cast<int>(o.x1);
    int label_y = static_cast<int>(o.y1);

    if (label_x + label_box_w > width) label_x = width - label_box_w;
    if (label_x < 0) label_x = 0;

    const int step_offset = label_box_h + 2;
    const int max_steps = 10;

    for (int s = 0; s < max_steps; ++s) {
        // 후보 라벨 사각형 (x1, y1, x2, y2)
        const int c0 = label_x, c1 = label_y, c2 = label_x + label_box_w, c3 = label_y + label_box_h;
        bool overlap = false;
        for (const auto& rr : drawn_label_rects) {
            const int r0 = rr.x, r1 = rr.y, r2 = rr.x + rr.width, r3 = rr.y + rr.height;
            if (!(c2 < r0 || c0 > r2 || c3 < r1 || c1 > r3)) { overlap = true; break; }
        }
        if (!overlap) break;
        label_y += step_offset;
        if (label_y + label_box_h > static_cast<int>(o.y2)) break;
    }

    if (label_y + label_box_h > height) label_y = height - label_box_h;

    const cv::Point tl(label_x, label_y), br(label_x + label_box_w, label_y + label_box_h);
    cv::rectangle(frame, tl, br, color, -1);                          // 라벨 배경
    cv::rectangle(frame, tl, br, cv::Scalar(255, 255, 255), 1);       // 흰색 테두리
    cv::putText(frame, display_text, cv::Point(label_x + pad_x, label_y + pad_y + text_h - 1),
                font, font_scale, text_color, font_thickness, cv::LINE_AA);

    drawn_label_rects.emplace_back(label_x, label_y, label_box_w, label_box_h);
}

// ==========================================
// 실행 시간 측정
// ==========================================
struct Timing {
    double read = 0, roi = 0, infer = 0, track = 0, risk = 0, draw = 0, show = 0, total = 0;
    long long frames = 0, objects = 0;
};

static double msSince(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static void printTiming(const Timing& t) {
    if (t.frames == 0) { std::cout << "[TIME] No frames measured.\n"; return; }
    const double n = static_cast<double>(t.frames);
    std::cout << std::fixed << std::setprecision(4)
              << "\n[TIME] Average per frame (" << t.frames << " frames, first frame excluded as warm-up)\n"
              << "  image read      : " << t.read / n  << " ms\n"
              << "  ROI setup/draw  : " << t.roi / n   << " ms\n"
              << "  YOLO inference  : " << t.infer / n << " ms  (preprocess + forward + NMS)\n"
              << "  tracking        : " << t.track / n << " ms\n"
              << "  risk algorithm  : " << t.risk / n  << " ms";
    if (t.objects > 0) std::cout << "  (" << (t.risk * 1000.0 / t.objects) << " us/object, " << t.objects << " objects)";
    std::cout << "\n"
              << "  drawing labels  : " << t.draw / n  << " ms\n"
              << "  display + wait  : " << t.show / n  << " ms\n"
              << "  TOTAL           : " << t.total / n << " ms  (" << (1000.0 * n / t.total) << " FPS)\n";
}

// ==========================================
// main
// ==========================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: fcw <best.onnx> <image_dir> [--no-display]\n";
        return 1;
    }
    const fs::path model_path = argv[1];
    const fs::path image_dir = argv[2];
    bool display = true;
    for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--no-display") display = false;

    std::vector<fs::path> image_files;
    if (fs::is_directory(image_dir)) {
        for (const auto& e : fs::directory_iterator(image_dir))
            if (e.is_regular_file() && lower(e.path().extension().string()) == ".jpg")
                image_files.push_back(e.path());
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cout << "[ERROR] No images found in: " << image_dir.string() << "\n";
        return 1;
    }

    std::unique_ptr<YoloOnnxDetector> detector;
    try {
        detector = std::make_unique<YoloOnnxDetector>(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[INFO] Starting processing of " << image_files.size() << " frames.\n";

    IouTracker tracker;
    std::unordered_map<int, std::deque<HistoryEntry>> track_history;
    Timing timing;
    bool warmup = true;
    const std::string window = "Monocular Camera-Based Forward Collision Prediction System";

    for (const auto& img_path : image_files) {
        const auto t_frame = Clock::now();
        Timing cur;

        // 이미지 읽기
        auto t = Clock::now();
        cv::Mat frame = readImage(img_path);
        cur.read = msSince(t);
        if (frame.empty()) continue;
        const int height = frame.rows, width = frame.cols;

        // 2-1. 주행 경로(ROI) 설정 및 중심점 도출
        t = Clock::now();
        const FrameContext ctx = buildFrameContext(width, height);
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{ctx.buffer_roi}, true, cv::Scalar(0, 255, 255), 1);
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{ctx.main_roi}, true, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, {ctx.cx_roi, ctx.cy_roi}, 6, cv::Scalar(255, 255, 255), -1);
        cv::circle(frame, {ctx.cx_roi, ctx.cy_roi}, 10, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "Center", {ctx.cx_roi + 15, ctx.cy_roi + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 2);
        cur.roi = msSince(t);

        // 2-2. YOLOv8 객체 검출 + 추적
        // Python 원본과 동일하게 ROI 오버레이가 그려진 프레임을 모델에 입력한다.
        t = Clock::now();
        std::vector<Detection> dets = detector->detect(frame);
        cur.infer = msSince(t);

        t = Clock::now();
        std::vector<TrackedObject> objects = tracker.update(dets);
        cur.track = msSince(t);

        // 2-3 ~ 2-6. 위험도 계산 (알고리즘 구간만 따로 측정)
        t = Clock::now();
        std::vector<RiskResult> risks;
        risks.reserve(objects.size());
        for (const auto& o : objects) risks.push_back(computeRisk(o, ctx, track_history));
        cur.risk = msSince(t);
        cur.objects = static_cast<long long>(objects.size());

        // 2-7. 시각화
        t = Clock::now();
        std::vector<cv::Rect> drawn_label_rects;
        for (size_t i = 0; i < objects.size(); ++i) drawObject(frame, objects[i], risks[i], drawn_label_rects);
        cur.draw = msSince(t);

        // 3. 영상 송출
        bool quit = false;
        t = Clock::now();
        if (display) {
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.5, 0.5);
            cv::imshow(window, resized);
            const int key = cv::waitKey(30) & 0xFF;
            if (key == 'q') {
                quit = true;
            } else if (key == ' ') {
                while (true) {
                    const int pause_key = cv::waitKey(0) & 0xFF;
                    if (pause_key == ' ') break;
                    if (pause_key == 'q') {
                        std::cout << "[EXIT] Program terminated by user.\n";
                        cv::destroyAllWindows();
                        printTiming(timing);
                        return 0;
                    }
                }
            }
        }
        cur.show = msSince(t);
        cur.total = msSince(t_frame);

        // 첫 프레임은 모델 초기화 비용이 섞이므로 통계에서 제외
        if (warmup) {
            warmup = false;
        } else {
            timing.read += cur.read;   timing.roi += cur.roi;   timing.infer += cur.infer;
            timing.track += cur.track; timing.risk += cur.risk; timing.draw += cur.draw;
            timing.show += cur.show;   timing.total += cur.total;
            timing.objects += cur.objects;
            ++timing.frames;
        }
        if (quit) break;
    }

    if (display) cv::destroyAllWindows();
    std::cout << "[DONE] Image sequence processing completed.\n";
    printTiming(timing);
    return 0;
}