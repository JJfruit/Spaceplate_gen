# quick_batch_fitness.py
import os, json, subprocess, sys, random
from pathlib import Path
from datetime import datetime

# Batch optimization runner

# === 사용자 설정 ===
MAIN_SCRIPT    = "gaussian_opt_write.py"   # ← 당신의 메인 스크립트명으로 바꾸세요
Z_R_TAG        = "zR_85um"                  # 로그가 쌓이는 폴더명 (예: zR_65um)
N_RUNS         = 50
FITNESS_TARGET = -1                      # 예) fitness ≥ -0.05 를 '성공'으로 간주

PY = sys.executable
ROOT = Path("ga_runs") / Z_R_TAG

def run_once(seed: int) -> Path | None:
    env = os.environ.copy()
    env["GA_SEED"] = str(seed)
    # 실행
    rc = subprocess.run([PY, MAIN_SCRIPT], env=env).returncode
    if rc != 0:
        return None
    # 가장 최근 run 디렉토리 찾기
    runs = [p for p in ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest

def read_best_fitness(run_dir: Path) -> float | None:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)
    return float(m.get("best_fitness")) if "best_fitness" in m else None

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    seeds = random.sample(range(10_000_000), N_RUNS)
    successes = 0
    values = []
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{N_RUNS}] seed={seed}")
        run_dir = run_once(seed)
        if run_dir is None:
            print("  -> run failed")
            continue
        bf = read_best_fitness(run_dir)
        if bf is None:
            print("  -> metrics.json or best_fitness missing")
            continue
        values.append(bf)
        ok = (bf >= FITNESS_TARGET)
        successes += int(ok)
        print(f"  best_fitness={bf:.6f}  ({'OK' if ok else 'FAIL'})")

    n = len(values)
    rate = (successes / n) if n else 0.0
    print("\n=== 결과 요약 ===")
    print(f"샘플 수: {n} / 시도: {N_RUNS}")
    print(f"성공 기준: best_fitness ≥ {FITNESS_TARGET}")
    print(f"성공 횟수: {successes}/{n}  → 성공률: {rate*100:.1f}%")

    if n:
        print(f"best_fitness 분포: min={min(values):.4f}, "
              f"median={sorted(values)[n//2]:.4f}, max={max(values):.4f}")

if __name__ == "__main__":
    main()
