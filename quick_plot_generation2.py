# quick_plot_runs_by_theta.py
import json
from pathlib import Path
from typing import Optional, Sequence, Dict, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import blended_transform_factory as _btf  # ★ 상단 정렬용

# =========================
# 사용자 설정
# =========================
ROOT = Path("ga_runs")

# 플롯에 포함할 run_id 목록
RUN_IDS = [
    "20251015-120017-f5b5cf","20251015-120429-e127c3","20251015-120849-0b2cce"
]

METRIC = "best"  # "best" | "mean" | "worst"
TITLE  = "GA fitness per generation (vs θ_max)"
YLIM   = (-80,5)

# --- y축 평행이동 옵션 ---
YOFFSET_MANUAL: Optional[float] = None
AUTO_SHIFT_TO_TOP: Optional[float] = None

# --- 수동 색상 지정 ---
COLORS: Dict[str, Union[str, Tuple[float,float,float]]] = {
    "20250926-051347-b0da60":"red",
    "20250926-053259-7ac215":"orange",
    "20250926-045137-05218b":"gray",
    "20250926-054829-e48e09":"green"
}
COLOR_LIST: Optional[Sequence[Union[str, Tuple[float,float,float]]]] = None

# --- 최대 fitness 위치 세로선 옵션 ---
SHOW_PEAK_VLINES: bool = True
PEAK_LINE_ALPHA: float = 0.6
PEAK_LINE_LS: str = "--"
PEAK_LINE_LW: float = 1.2
PEAK_MARKER: bool = True
PEAK_MARKER_SIZE: float = 30.0
PEAK_ANNOTATE: bool = True
PEAK_ANNOTATE_FMT: str = "fit={y:.2f}"

# ★ 주석을 플롯 상단(y=axes top)에 x값만 맞춰 정렬할지 여부
PEAK_ANNOTATE_AT_TOP: bool = True
PEAK_ANNOTATE_YOFFSET: float = -1

# --- θ 색상 매핑 범위(도 단위) ---
THETA_MIN: float = 0.0
THETA_MAX: float = 40.0

# =========================
# 내부 설정(색상/스타일)
# =========================
cmap = matplotlib.colormaps.get_cmap("viridis")
metric_col = {"best": "fitness_best", "mean": "fitness_mean", "worst": "fitness_worst"}[METRIC]
linestyle = {"best": "-", "mean": "--", "worst": ":"}[METRIC]

def _extract_theta_from_folder_name(name: str) -> Optional[float]:
    try:
        if name.startswith("THETA_") and name.endswith("deg"):
            core = name[len("THETA_"):-len("deg")]
            return float(core.replace("p", "."))
    except Exception:
        pass
    return None

def find_run_folder(root: Path, run_id: str) -> Optional[Path]:
    for tdir in root.glob("THETA_*deg"):
        cand = tdir / run_id
        if cand.is_dir():
            return cand
        for nadir in tdir.glob("NA_*"):
            cand2 = nadir / run_id
            if cand2.is_dir():
                return cand2
    for nadir in root.glob("NA_*"):
        cand = nadir / run_id
        if cand.is_dir():
            return cand
    hits: List[Path] = [p for p in root.rglob(run_id) if p.is_dir()]
    if hits:
        return hits[0]
    return None

def load_one(run_dir: Path):
    csv_path = run_dir / "fitness_per_generation.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    gen  = np.array(data["generation"], dtype=int)
    vals = np.array(data[metric_col], dtype=float)

    theta_max = None
    mpath = run_dir / "metrics.json"
    if mpath.exists():
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                theta_max = json.load(f).get("theta_max_deg_for_loss", None)
                if theta_max is not None:
                    theta_max = float(theta_max)
        except Exception:
            pass

    if theta_max is None:
        for parent in [run_dir.parent, run_dir.parent.parent]:
            if parent is None: continue
            th = _extract_theta_from_folder_name(parent.name)
            if th is not None:
                theta_max = th
                break

    return gen, vals, theta_max

def color_from_theta(theta_deg: Optional[float], idx: int, total: int):
    if theta_deg is None:
        t = 0.1 + 0.8 * (idx / max(1, total - 1))
        return cmap(t)
    t = (float(theta_deg) - THETA_MIN) / max(1e-9, (THETA_MAX - THETA_MIN))
    t = max(0.0, min(1.0, t))
    t = 0.1 + 0.8 * t
    return cmap(t)

def resolve_color(run_id: str, theta_deg: Optional[float], idx: int, total: int):
    if run_id in COLORS:
        return COLORS[run_id]
    if COLOR_LIST is not None and len(COLOR_LIST) == len(RUN_IDS):
        return COLOR_LIST[idx]
    return color_from_theta(theta_deg, idx, total)

def _series_label(run_id: str, theta_deg: Optional[float]) -> str:
    if theta_deg is not None:
        return f"(θ_max={theta_deg:.0f}°)"
    return run_id

def main():
    all_series = []
    global_max = None
    for rid in RUN_IDS:
        run_dir = find_run_folder(ROOT, rid)
        if run_dir is None: continue
        try:
            gen, val, theta_deg = load_one(run_dir)
        except Exception:
            continue
        all_series.append((rid, gen, val, theta_deg))
        vmax = float(np.nanmax(val))
        global_max = vmax if global_max is None else max(global_max, vmax)
    if not all_series: return

    offset = 0.0
    if YOFFSET_MANUAL is not None:
        offset = float(YOFFSET_MANUAL)
    elif AUTO_SHIFT_TO_TOP is not None and global_max is not None:
        offset = float(AUTO_SHIFT_TO_TOP) - float(global_max)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for i, (rid, gen, val, theta_deg) in enumerate(all_series):
        color = resolve_color(rid, theta_deg, i, len(all_series))
        label = _series_label(rid, theta_deg)
        y = val + offset
        ax.plot(gen, y, linestyle=linestyle, color=color, lw=1.8, label=label)

        if SHOW_PEAK_VLINES and np.isfinite(y).any():
            try:
                i_peak = int(np.nanargmax(val))
                g_peak = int(gen[i_peak]); y_peak = float(y[i_peak])
                ax.axvline(g_peak, color=color, ls=PEAK_LINE_LS, lw=PEAK_LINE_LW, alpha=PEAK_LINE_ALPHA)
                if PEAK_MARKER:
                    ax.scatter([g_peak], [y_peak], s=PEAK_MARKER_SIZE, color=color, edgecolors='white', zorder=3)
                if PEAK_ANNOTATE:
                    txt = PEAK_ANNOTATE_FMT.format(g=g_peak, y=y_peak)
                    if PEAK_ANNOTATE_AT_TOP:
                        trans = _btf(ax.transData, ax.transAxes)
                        extra_last = 0.06 if i == len(all_series) - 1 else 0.0  # 마지막 것만 올리기
                        ax.text(
                            g_peak, 1.0 + PEAK_ANNOTATE_YOFFSET + extra_last, txt,
                            transform=trans, ha='center', va='bottom',
                            fontsize=12, color=color,
                            bbox=dict(boxstyle="round",pad=0.2, fc="white", alpha=0.4, lw=0)
                        )
                    else:
                        ax.annotate(
                            txt, (g_peak, y_peak),
                            xytext=(5, 8 + (6 if i == len(all_series)-1 else 0)),
                            textcoords='offset points',
                            fontsize=9, color=color, ha='left', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.4, lw=0)
                        )
            except Exception:
                pass

    has_theta = any(s[3] is not None for s in all_series)
    if has_theta:
        norm = matplotlib.colors.Normalize(vmin=THETA_MIN, vmax=THETA_MAX)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.linspace(THETA_MIN, THETA_MAX, 256))
        cbar = plt.colorbar(sm, ax=ax, pad=0.015)
        cbar.set_label("θ_max (deg)", rotation=90)

    plt.title(TITLE, fontsize=18)
    plt.xlabel("Generation", fontsize=15)
    ylab = f"Fitness ({METRIC})"
    if offset != 0.0: ylab += f"  (offset {offset:+.1f})"
    plt.ylabel(ylab, fontsize=15)
    plt.grid(True, alpha=0.5)
    if YLIM is not None: plt.ylim(*YLIM)
    plt.legend(fontsize=12, loc="lower left")
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

