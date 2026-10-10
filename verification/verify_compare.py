# =============================================================
# [결과 동일성 검증 3단계] Python 결과와 C++ 결과를 한 줄씩 비교
#
# How to run : python verify_compare.py
# 입력 : py_risk.txt, cpp_risk.txt (같은 폴더)
# =============================================================

from collections import Counter

with open("py_risk.txt", encoding="utf-8") as fp:
    py = [line.strip() for line in fp if line.strip()]
with open("cpp_risk.txt", encoding="utf-8") as fp:
    cpp = [line.strip() for line in fp if line.strip()]

print(f"Python : {len(py)} values")
print(f"C++    : {len(cpp)} values")

mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(py, cpp)) if a != b]
if len(py) != len(cpp):
    print("[FAIL] 결과 개수가 다릅니다.")
elif mismatches:
    print(f"[FAIL] 불일치 {len(mismatches)}건 (처음 10건):")
    for i, a, b in mismatches[:10]:
        print(f"  #{i + 1}: Python={a}  C++={b}")
else:
    print(f"[PASS] {len(py)}건 모두 일치합니다.")

# 검증이 의미 있으려면 세 판정 상태가 고루 포함되어야 하므로 분포도 출력
def status(text):
    risk = int(text.split("Risk:")[1].rstrip("%"))
    return "DANGER" if risk >= 80 else "CAUTION" if risk > 40 else "SAFE"

dist = Counter(status(t) for t in py)
print(f"판정 분포 : SAFE {dist['SAFE']} / CAUTION {dist['CAUTION']} / DANGER {dist['DANGER']}")