# =============================================================
# Project : 생활도로(골목길)에서의 전방 추돌 예측 및 방지 알고리즘
# Team    : 김찬진, 양우영, 엄유빈, 이수홍
# Dataset : AI Hub "생활도로 객체인식 자율주행 데이터" (2023)
# Model   : YOLOv8m (Ultralytics, 2023)
#
# [시간 측정 버전] main_code.py와 판정 로직은 동일하고, C++ 버전(fcw.exe)과
# 같은 구간으로 실행 시간을 측정해 출력한다.
#
# How to run :
#   python main_code_timed.py <best.pt> <image_dir> [--no-display] [--device cpu|0]
#   --no-display : 화면 출력과 waitKey(30) 대기를 끄고 순수 처리 시간만 측정
#   --device     : 추론 장치 지정 (cpu 또는 GPU 번호 0). 생략하면 Ultralytics 자동 선택
# =============================================================

import argparse
import glob
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("image_dir")
parser.add_argument("--no-display", action="store_true")
parser.add_argument("--device", default=None)
args = parser.parse_args()

# ==========================================
# 1. 설정 및 경로 정의
# ==========================================
MODEL_PATH = args.model_path
IMAGE_DIR = args.image_dir
DISPLAY = not args.no_display

model = YOLO(MODEL_PATH)
image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
image_files.sort()

if not image_files:
    print(f"[ERROR] No images found in: {IMAGE_DIR}")
    exit()

print(f"[INFO] Starting processing of {len(image_files)} frames.")

track_history = {}
CONF_THRESHOLD = 0.5    # YOLO 객체 검출 신뢰도 임계값

# 하이퍼파라미터 (튜닝 요소)
FPS = 10                # 프레임 레이트 (10 FPS)
FRAME_INTERVAL = 10     # TTC/횡방향 분석 윈도우 (10프레임 ≈ 1초)

LABEL_NAMES = {0: "Pedestrian", 1: "Vehicle", 2: "PM"}

# ==========================================
# 실행 시간 측정 (C++ 버전과 같은 구간)
# ==========================================
STAGES = ["read", "roi", "infer_track", "risk", "draw", "show", "total"]
timing = {k: 0.0 for k in STAGES}
timing_frames = 0
timing_objects = 0
warmup = True


def ms_since(t):
    return (time.perf_counter() - t) * 1000.0


def print_timing():
    if timing_frames == 0:
        print("[TIME] No frames measured.")
        return
    n = timing_frames
    print(f"\n[TIME] Average per frame ({n} frames, first frame excluded as warm-up)")
    print(f"  image read      : {timing['read'] / n:.4f} ms")
    print(f"  ROI setup/draw  : {timing['roi'] / n:.4f} ms")
    print(f"  YOLO + tracking : {timing['infer_track'] / n:.4f} ms  (model.track, device={device_name})")
    line = f"  risk algorithm  : {timing['risk'] / n:.4f} ms"
    if timing_objects > 0:
        line += f"  ({timing['risk'] * 1000.0 / timing_objects:.4f} us/object, {timing_objects} objects)"
    print(line)
    print(f"  drawing labels  : {timing['draw'] / n:.4f} ms")
    print(f"  display + wait  : {timing['show'] / n:.4f} ms")
    print(f"  TOTAL           : {timing['total'] / n:.4f} ms  ({1000.0 * n / timing['total']:.4f} FPS)")


# --------------------------------------
# 2-3 ~ 2-6. 위험도 계산 (main_code.py 로직 그대로, 그리기만 분리)
# --------------------------------------
def compute_risk(box, obj_id, obj_cls, main_roi_points, buffer_roi_points,
                 cx_roi, cy_roi, max_dist, EGO_TURN_THRESHOLD, LATERAL_THRESHOLD):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    current_cx = int(x1 + w / 2)

    # 2-3. 판정점(ref_points) 설정 및 ROI 진입 판정
    # 클래스 ID(obj_cls): 0=Pedestrian, 1=Vehicle, 2=PM
    if obj_cls == 1:
        ref_points = [(int(x1 + w / 2), int(y2)), (int(x1), int(y1 + h / 2)), (int(x2), int(y1 + h / 2))]
    else:
        ref_points = [(int(x1), int(y2)), (int(x2), int(y2))]

    is_in_main = any(cv2.pointPolygonTest(main_roi_points, p, False) >= 0 for p in ref_points)
    is_in_buffer = any(cv2.pointPolygonTest(buffer_roi_points, p, False) >= 0 for p in ref_points)

    # 2-4. 데이터 누적 및 윈도우 기반 분석
    ttc_sec = float('inf')
    lateral_movement = 0
    is_moving_away = False

    if obj_id not in track_history:
        track_history[obj_id] = []

    track_history[obj_id].append({'area': area, 'cx': current_cx})

    if len(track_history[obj_id]) > FRAME_INTERVAL:
        track_history[obj_id].pop(0)

    if len(track_history[obj_id]) == FRAME_INTERVAL:
        prev_state = track_history[obj_id][0]
        prev_area = prev_state['area']
        prev_cx = prev_state['cx']

        if prev_area > 0:
            total_expansion_rate = (area - prev_area) / prev_area
            if total_expansion_rate > 0:
                rate_per_sec = (total_expansion_rate / FRAME_INTERVAL) * FPS
                ttc_sec = 2.0 / rate_per_sec

        lateral_movement = abs(current_cx - prev_cx)
        # ROI 무게중심으로부터 가로 방향으로 멀어지는지 판별
        if abs(current_cx - cx_roi) > abs(prev_cx - cx_roi):
            is_moving_away = True

    # 2-5. 정량적 위험도 계산
    # 1. 공간적 점수 (최대 40점): R_spatial = 20 + 20 * (1 - d/d_max)
    spatial_score = 0
    if is_in_main:
        valid_points = [p for p in ref_points if cv2.pointPolygonTest(main_roi_points, p, False) >= 0]
        if valid_points:
            min_dist = min([np.hypot(p[0] - cx_roi, p[1] - cy_roi) for p in valid_points])
            dist_ratio = max(0, min(1.0 - (min_dist / max_dist), 1.0))
            spatial_score = int(20 + (20 * dist_ratio))
        else:
            spatial_score = 20
    elif is_in_buffer:
        spatial_score = 20

    # 2. 시간적 점수 (최대 50점): TTC 역수 기반 비선형 정규화
    temporal_score = 0
    if obj_cls == 1:
        min_ttc, max_ttc = 1.5, 4.0     # 차량 : FCW 표준
    else:
        min_ttc, max_ttc = 2.5, 5.0     # 보행자/PM : Euro NCAP 기준

    if ttc_sec <= min_ttc:
        temporal_score = 50
    elif ttc_sec >= max_ttc:
        temporal_score = 0
    else:
        temporal_score = int(50 * ((1/ttc_sec - 1/max_ttc) / (1/min_ttc - 1/max_ttc)))

    if obj_cls == 1 and not is_in_main:
        temporal_score = 0

    # 3. 객체 가중치 (최대 10점): 교통약자 보호
    object_weight = 10 if (obj_cls == 0 or obj_cls == 2) else 0

    # 4. 최종 위험도 (0~100점)
    total_risk = min(100, spatial_score + temporal_score + object_weight)

    # 2-6. 횡방향 이탈 세분화
    discount_text = ""

    if lateral_movement > EGO_TURN_THRESHOLD:
        total_risk = int(total_risk * 0.2)
        discount_text = " (-Turn Ignore)"

    elif lateral_movement > LATERAL_THRESHOLD:
        if is_moving_away:
            total_risk = int(total_risk * 0.6)
            discount_text = " (-Pass By)"
        else:
            discount_text = " (!CUT-IN!)"

    return ref_points, total_risk


# --------------------------------------
# 2-7. 상태 판정 및 화면 시각화 (UI)
# --------------------------------------
def draw_object(frame, box, obj_cls, ref_points, total_risk, drawn_label_rects):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    for p in ref_points:
        cv2.circle(frame, p, 5, (255, 255, 255), -1)

    if total_risk >= 80:
        status, color, thickness = "DANGER", (0, 0, 255), 4      # 빨강
    elif total_risk > 40:
        status, color, thickness = "CAUTION", (0, 165, 255), 2   # 주황
    else:
        status, color, thickness = "SAFE", (255, 0, 0), 2        # 파랑

    # 바운딩 박스 그리기
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    obj_name = LABEL_NAMES.get(obj_cls, "Unknown")
    display_text = f"{obj_name}|Risk:{total_risk}%"

    # 다수 객체가 밀집한 골목길 환경에서 라벨이 겹치지 않도록
    # 박스 안쪽 좌상단에서 시작해 계단식으로 아래로 이동시키는 배치 로직
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)

    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

    pad_x, pad_y = 4, 3
    label_box_w = text_w + pad_x * 2
    label_box_h = text_h + pad_y * 2

    label_x = int(x1)
    label_y = int(y1)

    if label_x + label_box_w > width:
        label_x = width - label_box_w
    if label_x < 0:
        label_x = 0

    step_offset = label_box_h + 2
    max_steps = 10

    for _ in range(max_steps):
        cand_rect = (label_x, label_y, label_x + label_box_w, label_y + label_box_h)
        overlap = False
        for r in drawn_label_rects:
            if not (cand_rect[2] < r[0] or cand_rect[0] > r[2] or
                    cand_rect[3] < r[1] or cand_rect[1] > r[3]):
                overlap = True
                break
        if not overlap:
            break
        label_y += step_offset
        if label_y + label_box_h > int(y2):
            break

    if label_y + label_box_h > height:
        label_y = height - label_box_h

    cv2.rectangle(frame, (label_x, label_y), (label_x + label_box_w, label_y + label_box_h), color, -1)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_box_w, label_y + label_box_h), (255, 255, 255), 1)
    cv2.putText(frame, display_text, (label_x + pad_x, label_y + pad_y + text_h - 1),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    drawn_label_rects.append((label_x, label_y, label_x + label_box_w, label_y + label_box_h))


# ==========================================
# 2. 메인 처리 루프
# ==========================================
device_name = "auto"
for img_path in image_files:
    t_frame = time.perf_counter()
    cur = {k: 0.0 for k in STAGES}

    t = time.perf_counter()
    frame = cv2.imread(img_path)
    cur["read"] = ms_since(t)
    if frame is None:
        continue
    height, width = frame.shape[:2]

    # --------------------------------------
    # 2-1. 주행 경로(ROI) 설정 및 중심점 도출
    # --------------------------------------
    t = time.perf_counter()
    EGO_TURN_THRESHOLD = width * 0.1    # 자차 회전 판정 임계값 (화면 폭의 10% 이상 이동 시)
    LATERAL_THRESHOLD = width * 0.015   # 측면 객체 이동 판정 임계값 (화면 폭의 1.5% 이상 이동 시)

    main_roi_points = np.array([
        [int(width * 0.3), height],
        [int(width * 0.45), int(height * 0.65)],
        [int(width * 0.55), int(height * 0.65)],
        [int(width * 0.7), height]
    ], dtype=np.int32)

    buffer_roi_points = np.array([
        [int(width * 0.2), height],
        [int(width * 0.40), int(height * 0.65)],
        [int(width * 0.60), int(height * 0.65)],
        [int(width * 0.8), height]
    ], dtype=np.int32)

    cv2.polylines(frame, [buffer_roi_points], isClosed=True, color=(0, 255, 255), thickness=1)
    cv2.polylines(frame, [main_roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    M = cv2.moments(main_roi_points)
    if M['m00'] != 0:
        cx_roi = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
    else:
        cx_roi, cy_roi = int(width * 0.5), int(height * 0.8)

    cv2.circle(frame, (cx_roi, cy_roi), 6, (255, 255, 255), -1)
    cv2.circle(frame, (cx_roi, cy_roi), 10, (0, 0, 255), 2)
    cv2.putText(frame, "Center", (cx_roi + 15, cy_roi + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    max_dist = max([np.hypot(pt[0] - cx_roi, pt[1] - cy_roi) for pt in main_roi_points])
    cur["roi"] = ms_since(t)

    # --------------------------------------
    # 2-2. YOLOv8 객체 추적
    # --------------------------------------
    t = time.perf_counter()
    track_kwargs = dict(persist=True, conf=CONF_THRESHOLD, verbose=False)
    if args.device is not None:
        track_kwargs["device"] = args.device
    results = model.track(frame, **track_kwargs)
    boxes = ids = classes = None
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
    cur["infer_track"] = ms_since(t)
    if device_name == "auto":
        device_name = str(model.device) if hasattr(model, "device") else "auto"

    # --------------------------------------
    # 2-3 ~ 2-6. 위험도 계산 (알고리즘 구간만 따로 측정)
    # --------------------------------------
    t = time.perf_counter()
    risks = []
    if boxes is not None:
        for box, obj_id, obj_cls in zip(boxes, ids, classes):
            risks.append(compute_risk(box, obj_id, obj_cls, main_roi_points, buffer_roi_points,
                                      cx_roi, cy_roi, max_dist, EGO_TURN_THRESHOLD, LATERAL_THRESHOLD))
    cur["risk"] = ms_since(t)
    n_objects = len(risks)

    # --------------------------------------
    # 2-7. 시각화
    # --------------------------------------
    t = time.perf_counter()
    drawn_label_rects = []
    if boxes is not None:
        for box, obj_cls, (ref_points, total_risk) in zip(boxes, classes, risks):
            draw_object(frame, box, obj_cls, ref_points, total_risk, drawn_label_rects)
    cur["draw"] = ms_since(t)

    # --------------------------------------
    # 3. 영상 송출
    # --------------------------------------
    quit_now = False
    t = time.perf_counter()
    if DISPLAY:
        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Monocular Camera-Based Forward Collision Prediction System", resized_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            quit_now = True
        elif key == ord(' '):
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord(' '):
                    break
                elif pause_key == ord('q'):
                    print("[EXIT] Program terminated by user.")
                    cv2.destroyAllWindows()
                    print_timing()
                    exit()
    cur["show"] = ms_since(t)
    cur["total"] = ms_since(t_frame)

    # 첫 프레임은 모델 초기화 비용이 섞이므로 통계에서 제외
    if warmup:
        warmup = False
    else:
        for k in STAGES:
            timing[k] += cur[k]
        timing_objects += n_objects
        timing_frames += 1
    if quit_now:
        break

if DISPLAY:
    cv2.destroyAllWindows()
print("[DONE] Image sequence processing completed.")
print_timing()