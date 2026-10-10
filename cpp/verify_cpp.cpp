// =============================================================
// [결과 동일성 검증 2단계] 같은 시나리오를 C++ 위험도 로직에 입력
//
// main_code.cpp의 computeRisk()와 buildFrameContext()를 그대로 사용하고,
// YOLO 검출/추적 대신 scenario.csv의 박스를 넣어 위험도를 기록한다.
//
// Run : verify.exe scenario.csv cpp_risk.txt
// =============================================================

// main_code.cpp의 함수를 그대로 쓰기 위해 소스를 포함한다.
// (main 함수 이름만 바꿔서 이 파일의 main과 겹치지 않게 함)
#define main fcw_main
#include "main_code.cpp"
#undef main

#include <map>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: verify <scenario.csv> <cpp_risk.txt>\n";
        return 1;
    }

    // scenario.csv 한 줄: frame,id,cls,x1,y1,x2,y2
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "[ERROR] Cannot open " << argv[1] << "\n";
        return 1;
    }
    std::map<int, std::vector<TrackedObject>> frames;  // 프레임 순서대로 정렬
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> v;
        while (std::getline(ss, tok, ',')) v.push_back(tok);
        if (v.size() != 7) continue;
        TrackedObject o{std::strtof(v[3].c_str(), nullptr), std::strtof(v[4].c_str(), nullptr),
                        std::strtof(v[5].c_str(), nullptr), std::strtof(v[6].c_str(), nullptr),
                        std::stoi(v[1]), std::stoi(v[2])};
        frames[std::stoi(v[0])].push_back(o);
    }

    // Python 검증 코드와 같은 1920x1080 프레임 기준
    const FrameContext ctx = buildFrameContext(1920, 1080);
    std::unordered_map<int, std::deque<HistoryEntry>> track_history;

    std::ofstream out(argv[2]);
    long long count = 0;
    for (const auto& [frame_idx, objs] : frames) {
        for (const auto& o : objs) {
            const RiskResult r = computeRisk(o, ctx, track_history);
            const char* name = o.cls == PEDESTRIAN ? "Pedestrian" : o.cls == VEHICLE ? "Vehicle"
                             : o.cls == PM ? "PM" : "Unknown";
            out << name << "|Risk:" << r.total_risk << "%\n";
            ++count;
        }
    }
    std::cout << "[VERIFY] cpp_risk.txt : " << count << " risk values\n";
    return 0;
}