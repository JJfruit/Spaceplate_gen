# quick_plot_runs.py
import json
from pathlib import Path
from typing import Optional, Sequence, Dict, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import blended_transform_factory as _btf  # ★ 추가: 상단 정렬용

# =========================
# 사용자 설정
# =========================
ROOT = Path("ga_runs")
RUN_IDS = [
    "20251015-120017-f5b5cf","20251015-120429-e127c3","20251015-120849-0b2cce"
]
METRIC = "best"  # "best" | "mean" | "worst"
TITLE  = "GA fitness per generation (Gaussian Beam)"
YLIM   = (-8, 1)    # (-500, -100) 처럼 지정하거나 None

# --- y축 평행이동 옵션 ---
YOFFSET_MANUAL: Optional[float] = None
AUTO_SHIFT_TO_TOP: Optional[float] = None  # 예: -20.0 (전체 시리즈 최대값을 이 y값으로 끌어올림)

# --- 수동 색상 지정 옵션 ---
COLORS: Dict[str, Union[str, Tuple[float,float,float]]] = {
     "20250926-021954-23b9b3": "red",
     "20250926-133517-6f914f": "green",
     "20250928-102853-132e3e": "blue"
}
COLOR_LIST: Optional[Sequence[Union[str, Tuple[float,float,float]]]] = None  # RUN_IDS 순서대로 줄 색 (선택)

# --- 최대 fitness 위치 세로선 옵션 ---
SHOW_PEAK_VLINES: bool = True     # 세로선 On/Off
PEAK_LINE_ALPHA: float = 0.35     # 세로선 투명도
PEAK_LINE_LS: str = "--"          # 세로선 스타일 ("--", ":", "-.", ...)
PEAK_LINE_LW: float = 1.2         # 세로선 두께
PEAK_MARKER: bool = True          # 꼭짓점에 마커 표시 (원형 점)
PEAK_MARKER_SIZE: float = 30.0    # 마커 크기
PEAK_ANNOTATE: bool = True        # 세대 숫자 간단 주석
PEAK_ANNOTATE_FMT: str = "gen={g}"# 주석 포맷 (g: generation, y: fitness)

# ★ 추가: 주석을 플롯 상단(y=axes top)에 x값만 맞춰 정렬할지 여부
PEAK_ANNOTATE_AT_TOP: bool = True
PEAK_ANNOTATE_YOFFSET: float = -1  # 상단에서 얼마나 띄울지(axes fraction)

# =========================
# 내부 설정(색상/스타일)
# =========================
cmap = matplotlib.colormaps.get_cmap("viridis")
metric_col = {"best": "fitness_best", "mean": "fitness_mean", "worst": "fitness_worst"}[METRIC]
linestyle = {"best": "-", "mean": "--", "worst": ":"}[METRIC]

def find_run_folder(root: Path, run_id: str) -> Optional[Path]:
    for zdir in root.glob("zR_*um"):
        cand = zdir / run_id
        if cand.is_dir():
            return cand
    return None

def load_one(run_dir: Path):
    csv_path = run_dir / "fitness_per_generation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    gen  = np.array(data["generation"], dtype=int)
    vals = np.array(data[metric_col], dtype=float)

    zR_um = None
    mpath = run_dir / "metrics.json"
    if mpath.exists():
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                zR_um = json.load(f).get("z_R_um", None)
        except Exception:
            pass
    if zR_um is None:
        try:
            tag = run_dir.parent.name  # e.g., "zR_44um"
            if tag.startswith("zR_") and tag.endswith("um"):
                zR_um = float(tag[3:-2])
        except Exception:
            pass

    return gen, vals, zR_um

def color_from_zR(zR_um: Optional[float], idx: int, total: int):
    if zR_um is None:
        t = 0.1 + 0.8 * (idx / max(1, total - 1))
        return cmap(t)
    lo, hi = 10.0, 100.0
    t = (float(zR_um) - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    t = 0.1 + 0.8 * t
    return cmap(t)

def resolve_color(run_id: str, zR_um: Optional[float], idx: int, total: int):
    if run_id in COLORS:
        return COLORS[run_id]
    if COLOR_LIST is not None and len(COLOR_LIST) == len(RUN_IDS):
        return COLOR_LIST[idx]
    return color_from_zR(zR_um, idx, total)

def main():
    if not RUN_IDS:
        print("[힌트] RUN_IDS에 run_id를 넣어주세요.")
        return

    # 1) 시리즈 로딩 & 글로벌 최대(자동 오프셋용)
    all_series = []
    global_max = None
    for rid in RUN_IDS:
        run_dir = find_run_folder(ROOT, rid)
        if run_dir is None:
            print(f"[경고] 못 찾음: {rid}")
            continue
        try:
            gen, val, zR_um = load_one(run_dir)
        except Exception as e:
            print(f"[스킵] {rid}: {e}")
            continue
        all_series.append((rid, gen, val, zR_um))
        vmax = float(np.nanmax(val))
        global_max = vmax if global_max is None else max(global_max, vmax)

    if not all_series:
        print("[종료] 유효한 run이 없습니다.")
        return

    # 2) y-오프셋
    offset = 0.0
    if YOFFSET_MANUAL is not None:
        offset = float(YOFFSET_MANUAL)
    elif AUTO_SHIFT_TO_TOP is not None and global_max is not None:
        offset = float(AUTO_SHIFT_TO_TOP) - float(global_max)

    # 3) 플롯
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for i, (rid, gen, val, zR_um) in enumerate(all_series):
        color = resolve_color(rid, zR_um, i, len(all_series))
        label = f"(zR={zR_um:.0f} μm)" if zR_um is not None else rid

        y = val + offset
        ax.plot(gen, y, linestyle=linestyle, color=color, lw=1.8, label=label)

        # --- 최대 fitness 지점 표시 ---
        if SHOW_PEAK_VLINES and np.isfinite(y).any():
            try:
                i_peak = int(np.nanargmax(val))    # 원래 값 기준으로 peak index
                g_peak = int(gen[i_peak])
                y_peak = float(y[i_peak])

                # 세로선
                ax.axvline(g_peak, color=color, ls=PEAK_LINE_LS, lw=PEAK_LINE_LW, alpha=PEAK_LINE_ALPHA)

                # 마커 (선택)
                if PEAK_MARKER:
                    ax.scatter([g_peak], [y_peak], s=PEAK_MARKER_SIZE, color=color, edgecolors='white', zorder=3)

                # 간단 주석 (선택)
                if PEAK_ANNOTATE:
                    txt = PEAK_ANNOTATE_FMT.format(g=g_peak, y=y_peak)
                    if PEAK_ANNOTATE_AT_TOP:
                        # x는 data 좌표, y는 axes 비율 좌표(1.0이 상단)
                        trans = _btf(ax.transData, ax.transAxes)
                        ax.text(
                            g_peak, 1.0 + PEAK_ANNOTATE_YOFFSET, txt,
                            transform=trans, ha='center', va='bottom',
                            fontsize=12, color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.4, lw=0)
                        )
                    else:
                        # 기존: 점 근처에 붙이기
                        ax.annotate(
                            txt, (g_peak, y_peak),
                            xytext=(5, 8), textcoords='offset points',
                            fontsize=9, color=color, ha='left', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.4, lw=0)
                        )
            except Exception:
                pass

    plt.title(TITLE, fontsize=18)
    plt.xlabel("Generation", fontsize=15)
    ylab = f"Fitness ({METRIC})"
    plt.ylabel(ylab, fontsize=15)
    plt.grid(True, alpha=0.5)
    if YLIM is not None:
        plt.ylim(*YLIM)
    plt.legend(fontsize=12, loc="lower left")
    plt.tick_params(axis='both',labelsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

