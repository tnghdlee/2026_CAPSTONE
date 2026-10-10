# =============================================================
# [결과 동일성 검증 1단계] 시나리오 생성 + 원본 Python 코드 실행
#
# YOLO 검출/추적을 빼고, 무작위로 만든 박스 좌표를 원본 코드의
# 위험도 로직에 그대로 넣어 결과(위험도 문자열)를 기록한다.
#
# How to run : python verify_python.py main_code.py
#   - 인자로 원본 캡스톤 코드(Python) 경로를 넣는다.
#   - 원본 코드는 수정하지 않고 그대로 실행한다. YOLO 모델, 이미지 읽기,
#     화면 출력만 가짜로 바꿔 끼운다.
# 출력 : scenario.csv (C++에 넣을 박스), py_risk.txt (Python 위험도 결과)
# =============================================================

import glob
import random
import sys
import types

import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python verify_python.py <main_code.py>")
    sys.exit(1)
SRC_PATH = sys.argv[1]

W, H, NUM_FRAMES = 1920, 1080, 400   # C++ 검증 코드도 같은 해상도를 사용
SEED = 7                             # 같은 시드면 항상 같은 시나리오가 만들어짐

# ------------------------------------------
# 1. 무작위 추적 시나리오 생성
#    객체 25개: 다가오기(박스 확대), 멀어지기, 좌우 이동, 갑작스러운 점프(회전 상황)
# ------------------------------------------
rng = random.Random(SEED)
objects = []
for oid in range(1, 26):
    cls = rng.choice([0, 1, 1, 2])   # 0=Pedestrian, 1=Vehicle, 2=PM
    start = rng.randrange(0, NUM_FRAMES - 30)
    end = min(NUM_FRAMES, start + rng.randrange(15, 120))
    objects.append(dict(
        id=oid, cls=cls, start=start, end=end,
        cx=rng.uniform(200, 1700), by=rng.uniform(650, 1080), bw=rng.uniform(40, 300),
        vx=rng.choice([0, 0, rng.uniform(-25, 25)]),     # 좌우 이동 속도
        g=rng.uniform(0.97, 1.08),                       # 프레임당 크기 변화율
        jump=rng.random() < 0.2,                         # 가끔 크게 튀는 객체
    ))

frames = {f: [] for f in range(NUM_FRAMES)}
for o in objects:
    cx, by, bw = o['cx'], o['by'], o['bw']
    for f in range(o['start'], o['end']):
        cx += o['vx'] + rng.uniform(-3, 3)
        bw *= o['g']
        by = min(H, by + (o['g'] - 1) * 40)
        if o['jump'] and rng.random() < 0.05:
            cx += rng.choice([-1, 1]) * 260
        bw = max(10, min(bw, 900))
        bh = bw * (2.2 if o['cls'] == 0 else 0.8)
        box = np.array([cx - bw / 2, by - bh, cx + bw / 2, by], dtype=np.float32)
        box = np.clip(box, [0, 0, 0, 0], [W, H, W, H]).astype(np.float32)
        frames[f].append((o['id'], o['cls'], box))

with open("scenario.csv", "w", encoding="utf-8") as fp:
    for f in range(NUM_FRAMES):
        for oid, cls, box in frames[f]:
            fp.write(f"{f},{oid},{cls}," + ",".join(repr(float(v)) for v in box) + "\n")

# ------------------------------------------
# 2. 원본 코드가 쓰는 외부 요소만 가짜로 교체
# ------------------------------------------
class _T:
    def __init__(self, a): self.a = a
    def cpu(self): return self
    def numpy(self): return self.a

class _Boxes:
    def __init__(self, rows):
        self.xyxy = _T(np.array([r[2] for r in rows], np.float32).reshape(-1, 4))
        self.id = _T(np.array([r[0] for r in rows], np.float32)) if rows else None
        self.cls = _T(np.array([r[1] for r in rows], np.float32))

class _Result:
    def __init__(self, rows): self.boxes = _Boxes(rows)

class FakeYOLO:
    """model.track()이 호출될 때마다 시나리오의 다음 프레임 박스를 돌려준다."""
    device = "fake"
    def __init__(self, *args, **kwargs): self.frame = 0
    def track(self, *args, **kwargs):
        rows = frames.get(self.frame, [])
        self.frame += 1
        return [_Result(rows)]

fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake_ultralytics

fake_files = [f"frame_{i:05d}.jpg" for i in range(NUM_FRAMES)]
glob.glob = lambda *args, **kwargs: list(fake_files)              # 이미지 목록
blank = np.zeros((H, W, 3), np.uint8)
cv2.imread = lambda *args, **kwargs: blank.copy()                  # 빈 프레임
cv2.imshow = lambda *args, **kwargs: None                          # 화면 출력 끄기
cv2.waitKey = lambda *args, **kwargs: -1
cv2.destroyAllWindows = lambda *args, **kwargs: None

risk_texts = []
_real_put_text = cv2.putText
def _capture_put_text(img, text, *args, **kwargs):
    if "|Risk:" in text:            # 객체 라벨("Vehicle|Risk:37%")만 기록
        risk_texts.append(text)
    return _real_put_text(img, text, *args, **kwargs)
cv2.putText = _capture_put_text

# ------------------------------------------
# 3. 원본 코드 실행
# ------------------------------------------
sys.argv = [SRC_PATH, "fake.pt", "fake_dir", "--no-display"]   # 시간 측정 버전 호환용
with open(SRC_PATH, encoding="utf-8") as fp:
    source = fp.read()
exec(compile(source, SRC_PATH, "exec"), {"__name__": "__main__"})

with open("py_risk.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(risk_texts) + "\n")

print(f"\n[VERIFY] scenario.csv : {sum(len(v) for v in frames.values())} boxes, {NUM_FRAMES} frames")
print(f"[VERIFY] py_risk.txt  : {len(risk_texts)} risk values")