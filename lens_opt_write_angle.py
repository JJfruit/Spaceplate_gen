import os, json, time, uuid, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from tmm_core_test import coh_tmm
import pygad

# GA optimizer for lens systems

# =========================
# 0) 실행 로깅/저장 유틸 (NA 또는 θ 기준)
# =========================
LOG_ROOT = Path("ga_runs")

def _na_tag(NA: float) -> str:
    # 예: 0.1 -> "NA_0p10"
    return "NA_" + f"{NA:.3f}".replace(".", "p").rstrip("0").rstrip("p")

def _theta_tag(theta_deg: float) -> str:
    # 예: 10.0 -> "THETA_10p0deg"
    return "THETA_" + f"{theta_deg:.1f}".replace(".", "p") + "deg"

def _make_run_dir(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + "-" + uuid.uuid4().hex[:6]
    run_dir = base_dir / run_id
    run_dir.mkdir(exist_ok=True)
    return base_dir, run_dir, run_id

def make_run_paths_by_NA(NA: float):
    # (하위 호환용) NA 기준 기본 경로
    na_dir = LOG_ROOT / _na_tag(NA)
    na_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + "-" + uuid.uuid4().hex[:6]
    run_dir = na_dir / run_id
    run_dir.mkdir(exist_ok=True)
    return na_dir, run_dir, run_id

class RunLogger:
    def __init__(self, NA, note="", meta=None, group_by="NA", theta_deg=None):
        """
        group_by: "NA" | "THETA" | "BOTH"
          - "NA":    ga_runs/NA_.../run_id
          - "THETA": ga_runs/THETA_.../run_id
          - "BOTH":  ga_runs/THETA_.../NA_.../run_id
        """
        self.NA = float(NA)
        self.theta_deg = float(theta_deg) if theta_deg is not None else None
        self.group_by = (group_by or "NA").upper()

        if self.group_by == "THETA":
            base_dir = LOG_ROOT / _theta_tag(self.theta_deg)
        elif self.group_by == "BOTH":
            assert self.theta_deg is not None, "group_by='BOTH'는 theta_deg 필요"
            base_dir = LOG_ROOT / _theta_tag(self.theta_deg) / _na_tag(self.NA)
        else:  # "NA"
            base_dir = LOG_ROOT / _na_tag(self.NA)

        # 기존 변수명 유지(na_dir) — 이제는 '그룹 디렉토리' 의미
        self.na_dir, self.run_dir, self.run_id = _make_run_dir(base_dir)

        self.note = note
        self.meta = meta or {}
        self.fitness_rows = []  # (gen, best, mean, worst)
        self.t0 = time.time()

        meta_obj = {
            "run_id": self.run_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "group_by": self.group_by,
            "NA": self.NA,
            "theta_max_deg_for_loss": self.theta_deg,
            "note": note,
            **self.meta
        }
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    def on_generation(self, ga):
        gen = ga.generations_completed
        pop_fit = getattr(ga, "last_generation_fitness", None)
        if pop_fit is None:
            def eval_one(sol):
                return ga.fitness_func(sol, None)
            pop_fit = np.array([eval_one(ind) for ind in ga.population])
        best = float(np.max(pop_fit))
        mean = float(np.mean(pop_fit))
        worst = float(np.min(pop_fit))
        self.fitness_rows.append((gen, best, mean, worst))

    def flush_fitness_csv(self):
        csv_path = self.run_dir / "fitness_per_generation.csv"
        header = ["generation", "fitness_best", "fitness_mean", "fitness_worst"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.fitness_rows)

    def save_best_d_list(self, best_solution_nm_list):
        with open(self.run_dir / "best_d_list.json", "w", encoding="utf-8") as f:
            json.dump({"best_d_list_nm": list(map(float, best_solution_nm_list))}, f, indent=2)

    def save_metrics(self, metrics_dict):
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        # 요약 CSV는 '그룹 디렉토리' 바로 아래에 누적
        summary_csv = self.na_dir / "runs_summary.csv"
        header = ["run_id","timestamp","group_by","theta_max_deg_for_loss","NA",
                  "compression_ratio_target","R_best","total_thickness_um",
                  "T_rms_best","phase_mse_space","phase_mse_angle",
                  "gen_at_best","gens_total","minutes_elapsed","note"]
        file_exists = summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow([
                self.run_id,
                time.strftime('%Y-%m-%d %H:%M:%S'),
                self.group_by,
                metrics_dict.get("theta_max_deg_for_loss"),
                metrics_dict.get("NA"),
                metrics_dict.get("compression_ratio_target"),
                metrics_dict.get("R_best"),
                metrics_dict.get("total_thickness_um"),
                metrics_dict.get("T_rms_best"),
                metrics_dict.get("phase_mse_space"),
                metrics_dict.get("phase_mse_angle"),
                metrics_dict.get("gen_at_best"),
                metrics_dict.get("gens_total"),
                metrics_dict.get("minutes_elapsed"),
                self.note
            ])

    def elapsed_minutes(self):
        return (time.time() - self.t0) / 60.0


# =========================
# 원래 코드 (연산 변경 없음, NA/θ 로깅만 추가)
# =========================

# --- 파라미터 ---
wavelength = 1550e-9
k0 = 2 * np.pi / wavelength
Nx = 128
x = np.linspace(-1e-3, 1e-3, Nx)
dx = x[1] - x[0]
f_lens = 10e-3
z_gap = 1e-3
z_target = 50e-6     # d_eff

# --- NA 입력 (여기만 바꾸면 구경 자동 조정; f_lens는 고정) ---
NA = 0.10  # 예시값: 0.10
# 공기(n=1)에서 NA = sin(alpha). 구경 반지름 a = f * tan(alpha) = f * NA / sqrt(1-NA^2)
_aperture_radius = f_lens * NA / np.sqrt(max(1e-16, 1.0 - NA**2))
if _aperture_radius > np.max(np.abs(x)):
    print(f"[Warn] aperture radius ({_aperture_radius*1e3:.2f} mm)가 x 범위(±{np.max(np.abs(x))*1e3:.2f} mm)를 초과합니다. "
          "필요시 x 범위를 넓혀 주세요.")

# --- 기준값 설정 ---
intensity_threshold_ratio = 0.5  # 목표 intensity 비율 (T_rms 하한)
phase_mse_threshold       = 1e-3 # phase MSE 컷(공간 위상)
angle_mse_threshold       = 1e-1
weight_intensity          = 0.1  # intensity penalty 가중치
compression_ratio         = 5

# === 추가: 손실 가중치 (공간위상 vs 각도위상) ===
W_SPACE = 0.1     # 공간(마스크) 위상 MSE 가중치
W_ANGLE = 4.0     # 각도 위상 MSE 가중치

# === 추가: 마스크/정렬 옵션 ===
intensity_mask_ratio = 0.02      # 레퍼런스 강도 상위 2% 이상 구간만 사용 권장
CENTER_ALIGN_PHASE   = True       # (공간) 중심 위상 정렬 여부

# === 추가: 각도 위상 비교 범위/샘플 수 ===
theta_max_deg_for_loss = 20
N_theta_loss = 181  # 0~theta_max 범위에서 균등 샘플

# --- 굴절률 구조 생성 ---
def alternating_n_list(N):
    n1, n2 = 3.6211 + 1.26e-7j, 1.4596 + 0.000011139j
    return [n1 if i % 2 == 0 else n2 for i in range(N)]

# --- ASM 전파 함수 ---
def angular_spectrum_1d(E_in, wavelength, distance, dx, n_medium=1.0):
    fx = np.fft.fftfreq(len(E_in), d=dx)
    kx = 2 * np.pi * fx
    k0 = 2 * np.pi / wavelength
    kz = np.sqrt((n_medium * k0)**2 - kx**2 + 0j)
    H = np.exp(1j * kz * distance)
    E_fft = fft(E_in)
    E_fft_prop = E_fft * H
    return ifft(E_fft_prop)

# --- 레퍼런스 필드 계산 ---
def calculate_ref_output():
    R2 = x**2
    phi = -k0*(np.sqrt(R2+f_lens**2)-f_lens)
    # aperture: NA로부터 결정된 반지름 사용
    aperture = R2 < (_aperture_radius**2)
    incident_phase = np.exp(1j * k0 * x * np.sin(0))
    E = aperture * incident_phase
    E *= np.exp(1j * phi)
    E_after_gap = angular_spectrum_1d(E, wavelength, z_gap, dx, n_medium=1.0)
    E_ref = angular_spectrum_1d(E_after_gap, wavelength, z_target, dx, n_medium=1.0)
    return E_after_gap, E_ref

E_after_gap, E_ref_field = calculate_ref_output()
phase_ref_all = np.unwrap(np.angle(E_ref_field))
I_ref         = np.abs(E_ref_field)**2
I_ref_max     = np.max(I_ref)

# === 레퍼런스 강도 마스크 (z_target 면에서 생성) ===
mask = I_ref >= (intensity_mask_ratio * I_ref_max)
if not np.any(mask):
    mask = np.zeros_like(I_ref, dtype=bool)
    mask[Nx//2] = True

# --- 구조체 시뮬레이션 함수 ---
def run_simulation_from_E(E_input, n_list, d_list, pol='s'):
    fx = np.fft.fftfreq(Nx, d=dx)
    kx = 2 * np.pi * fx
    theta_kx = np.arcsin(np.clip(kx / k0, -1, 1))
    t_coef_array = np.zeros(Nx, dtype=complex)

    for i, theta in enumerate(theta_kx):
        try:
            tmm_result = coh_tmm(pol, n_list, d_list, theta, wavelength*1e9)  # wavelength in nm
            t_coef_array[i] = tmm_result['t']
        except:
            t_coef_array[i] = 0.0

    E_fft = fft(E_input)
    E_fft *= t_coef_array
    return ifft(E_fft)

# --- 각도 평균 RMS Transmission ---
def compute_rms_transmission(n_list, d_list, pol='s', angle_range=np.linspace(-15, 15, 64), wavelength=1550):
    T_array = []
    for theta_deg in angle_range:
        theta_rad = np.deg2rad(theta_deg)
        try:
            tmm_result = coh_tmm(pol, n_list, d_list, theta_rad, wavelength)  # wavelength in nm
            T_array.append(tmm_result['T'])
        except:
            T_array.append(0.0)
    T_array = np.array(T_array)
    return np.sqrt(np.mean(T_array**2))

# === 각도 위상 계산 유틸 ===
def transmission_phase_vs_theta(n_list, d_list, pol, theta_deg_array, lam_vac_nm):
    phases = np.zeros_like(theta_deg_array, dtype=float)
    for i, th in enumerate(theta_deg_array):
        try:
            th_rad = np.deg2rad(th)
            res = coh_tmm(pol, n_list, d_list, th_rad, lam_vac_nm)
            phases[i] = np.angle(res['t'])
        except:
            phases[i] = np.nan
    phases = np.unwrap(phases)
    return phases

# === 각도 위상 레퍼런스(자유공간 z_target) 미리 계산 ===
theta_deg_array_loss = np.linspace(0.0, theta_max_deg_for_loss, N_theta_loss)
n_list_free = [1.0, 1.0, 1.0]
d_list_free = [np.inf, z_target*1e9, np.inf]

phi_free_s_ref = transmission_phase_vs_theta(n_list_free, d_list_free, 's',
                                             theta_deg_array_loss, wavelength*1e9)
phi_free_p_ref = transmission_phase_vs_theta(n_list_free, d_list_free, 'p',
                                             theta_deg_array_loss, wavelength*1e9)
phi_free_s_ref -= phi_free_s_ref[0]
phi_free_p_ref -= phi_free_p_ref[0]

# --- Fitness 함수 ---
def fitness_func(ga_instance, solution, solution_idx):
    d_solution = list(solution)
    d_list = [np.inf] + d_solution + [np.inf]
    n_list = [1.0] + alternating_n_list(len(d_solution)) + [1.0]

    total_thickness_m = sum(d_solution) * 1e-9
    R = z_target / total_thickness_m
    if R < compression_ratio:
        return -1e5

    T_rms_avg = 0.5 * (compute_rms_transmission(n_list, d_list, 's') +
                       compute_rms_transmission(n_list, d_list, 'p'))
    if T_rms_avg < intensity_threshold_ratio:
        return -1e4

    E_s_exit = run_simulation_from_E(E_after_gap, n_list, d_list, pol='s')
    E_p_exit = run_simulation_from_E(E_after_gap, n_list, d_list, pol='p')

    phase_s_all = np.unwrap(np.angle(E_s_exit))
    phase_p_all = np.unwrap(np.angle(E_p_exit))
    phase_r_all = phase_ref_all.copy()

    if CENTER_ALIGN_PHASE:
        c = Nx // 2
        phase_r_all -= phase_r_all[c]
        phase_s_all -= phase_s_all[c]
        phase_p_all -= phase_p_all[c]

    loss_phase_space = 0.5 * (
        np.mean((phase_r_all[mask] - phase_s_all[mask])**2) +
        np.mean((phase_r_all[mask] - phase_p_all[mask])**2)
    )

    phi_plate_s = transmission_phase_vs_theta(n_list, d_list, 's',
                                              theta_deg_array_loss, wavelength*1e9)
    phi_plate_p = transmission_phase_vs_theta(n_list, d_list, 'p',
                                              theta_deg_array_loss, wavelength*1e9)
    phi_plate_s -= phi_plate_s[0]
    phi_plate_p -= phi_plate_p[0]

    loss_phase_theta = 0.5 * (
        np.nanmean((phi_free_s_ref - phi_plate_s)**2) +
        np.nanmean((phi_free_p_ref - phi_plate_p)**2)
    )

    if loss_phase_theta > angle_mse_threshold:
        return -(1e2 * loss_phase_theta)

    intensity_error = max(0, intensity_threshold_ratio - T_rms_avg)
    total_loss = (W_SPACE * loss_phase_space) + (W_ANGLE * loss_phase_theta) \
                 + weight_intensity * intensity_error

    return -total_loss

# --- 콜백 + 로거 연결 ---
_last_fit = None
logger = RunLogger(
    NA=NA,
    note="lens spherical wave optimization",
    meta={
        "wavelength_nm": float(wavelength*1e9),
        "f_lens_mm": float(f_lens*1e3),
        "z_gap_mm": float(z_gap*1e3),
        "z_target_um": float(z_target*1e6),
        "aperture_radius_mm": float(_aperture_radius*1e3),
        "Nx": Nx, "dx_um": float(dx*1e6),
        "intensity_threshold_ratio": intensity_threshold_ratio,
        "phase_mse_threshold": phase_mse_threshold,
        "angle_mse_threshold": angle_mse_threshold,
        "W_SPACE": W_SPACE, "W_ANGLE": W_ANGLE,
        "theta_max_deg_for_loss": theta_max_deg_for_loss,
        "N_theta_loss": N_theta_loss,
        "compression_ratio_target": compression_ratio
    },
    group_by="THETA",                 # ← 저장 기준: "NA" | "THETA" | "BOTH"
    theta_deg=theta_max_deg_for_loss  # ← group_by에 "THETA"/"BOTH"일 때 필요
)

def on_generation(ga_instance):
    global _last_fit
    current_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )[1]
    delta = 0 if _last_fit is None else current_fitness - _last_fit
    print(f"[Gen {ga_instance.generations_completed:02d}] Best Fit: {current_fitness:.6f}  Δ:{delta:+.6f}")
    _last_fit = current_fitness
    logger.on_generation(ga_instance)

# --- GA 실행 ---
ga_instance = pygad.GA(
    num_generations=500,
    sol_per_pop=500,
    num_parents_mating=250,
    num_genes=25,
    gene_space={'low': 10, 'high': 1200},  # [nm]
    fitness_func=fitness_func,
    mutation_type="random",
    crossover_type="single_point",
    mutation_percent_genes=20,
    keep_elitism=4,
    on_generation=on_generation
)

ga_instance.run()
logger.flush_fitness_csv()

# --- 결과 출력 및 R 값 ---
best_solution, best_fitness, _ = ga_instance.best_solution()
print("Best d_list (nm):", ', '.join(map(str, best_solution)))
total_thickness_m = sum(best_solution) * 1e-9
R_best = z_target / total_thickness_m
print(f"Compression ratio R: {R_best:.2f}")
sum_nm = float(np.sum(best_solution))
print(f"Total stack thickness: {sum_nm:.2f} nm ({total_thickness_m*1e6:.3f} µm)")

# --- 베스트 해 재평가 ---
n_list_opt = [1.0] + alternating_n_list(len(best_solution)) + [1.0]
d_list_opt = [np.inf] + list(best_solution) + [np.inf]

E_s_best = run_simulation_from_E(E_after_gap, n_list_opt, d_list_opt, pol='s')
E_p_best = run_simulation_from_E(E_after_gap, n_list_opt, d_list_opt, pol='p')

phase_s_best_all = np.unwrap(np.angle(E_s_best))
phase_p_best_all = np.unwrap(np.angle(E_p_best))

c = Nx // 2
phase_ref_eval = phase_ref_all - phase_ref_all[c] if CENTER_ALIGN_PHASE else phase_ref_all.copy()
phase_s_eval   = phase_s_best_all - phase_s_best_all[c] if CENTER_ALIGN_PHASE else phase_s_best_all.copy()
phase_p_eval   = phase_p_best_all - phase_p_best_all[c] if CENTER_ALIGN_PHASE else phase_p_best_all.copy()

phase_mse_best = 0.5 * (
    np.mean((phase_ref_eval[mask] - phase_s_eval[mask])**2) +
    np.mean((phase_ref_eval[mask] - phase_p_eval[mask])**2)
)
print(f"Best Phase MSE (space, masked, avg s/p): {phase_mse_best:.6e}")

T_rms_best = 0.5 * (
    compute_rms_transmission(n_list_opt, d_list_opt, 's') +
    compute_rms_transmission(n_list_opt, d_list_opt, 'p')
)
print(f"Best Intensity RMS (avg s/p): {T_rms_best:.4f}")

theta_deg_array_loss = np.linspace(0.0, theta_max_deg_for_loss, N_theta_loss)
phi_plate_s_best = transmission_phase_vs_theta(n_list_opt, d_list_opt, 's',
                                               theta_deg_array_loss, wavelength*1e9)
phi_plate_p_best = transmission_phase_vs_theta(n_list_opt, d_list_opt, 'p',
                                               theta_deg_array_loss, wavelength*1e9)
phi_plate_s_best -= phi_plate_s_best[0]
phi_plate_p_best -= phi_plate_p_best[0]

theta_phase_mse_best = 0.5 * (
    np.nanmean((phi_free_s_ref - phi_plate_s_best)**2) +
    np.nanmean((phi_free_p_ref - phi_plate_p_best)**2)
)
print(f"Best Phase MSE (angle, 0–{theta_max_deg_for_loss:.0f}°, avg s/p): {theta_phase_mse_best:.6e}")

# --- gen@best / 요약 저장 ---
if logger.fitness_rows:
    gens = np.array([g for g,_,_,_ in logger.fitness_rows], dtype=int)
    bests = np.array([b for _,b,_,_ in logger.fitness_rows], dtype=float)
    gen_at_best = int(gens[np.argmax(bests)])
    gens_total = int(gens.max()) if gens.size else int(ga_instance.generations_completed)
else:
    gen_at_best = int(ga_instance.generations_completed)
    gens_total = int(ga_instance.generations_completed)

metrics = {
    "run_id": logger.run_id,
    "NA": float(NA),
    "theta_max_deg_for_loss": float(theta_max_deg_for_loss),  # ← θ 기록
    "compression_ratio_target": compression_ratio,
    "R_best": float(R_best),
    "total_thickness_um": float(total_thickness_m*1e6),
    "T_rms_best": float(T_rms_best),
    "phase_mse_space": float(phase_mse_best),
    "phase_mse_angle": float(theta_phase_mse_best),
    "gen_at_best": gen_at_best,
    "gens_total": gens_total,
    "minutes_elapsed": float(logger.elapsed_minutes())
}
logger.save_metrics(metrics)
logger.save_best_d_list(best_solution)

print(f"[Saved] Run folder: {logger.run_dir}")
print(f"[Saved] Summary appended to: {logger.na_dir / 'runs_summary.csv'}")

# --- (이하는 시각화: 원 코드 유지) ---
theta_deg_array = np.linspace(-90, 90, 1000)
phase_opt   = np.zeros_like(theta_deg_array)
phase_free  = np.zeros_like(theta_deg_array)

for i, theta_deg in enumerate(theta_deg_array):
    theta_rad = np.deg2rad(theta_deg)
    try:
        phase_opt[i]  = np.angle(coh_tmm('s', n_list_opt,  d_list_opt,  theta_rad, wavelength*1e9)['t'])
        phase_free[i] = np.angle(coh_tmm('s', [1.0,1.0,1.0], [np.inf, z_target*1e9, np.inf],
                                         theta_rad, wavelength*1e9)['t'])
    except:
        phase_opt[i] = phase_free[i] = np.nan

phase_opt  = np.unwrap(phase_opt)  - np.max(np.unwrap(phase_opt))
phase_free = np.unwrap(phase_free) - np.max(np.unwrap(phase_free))

plt.figure(figsize=(8,5))
plt.plot(theta_deg_array, phase_opt,  label='Optimized Plate')
plt.plot(theta_deg_array, phase_free, '--', label='Free Space')
plt.xlim(0, 40)
plt.ylim(-10, 0.5)
plt.xlabel('Incident angle (deg)')
plt.ylabel('Transmission phase (rad)')
plt.title('Phase: Plate vs Free Space (wide view)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(theta_deg_array_loss, phi_plate_s_best, label='Plate φ_s(θ) [0-aligned]')
plt.plot(theta_deg_array_loss, phi_free_s_ref, '--', label='Free φ_s(θ) [0-aligned]')
plt.plot(theta_deg_array_loss, phi_plate_p_best, label='Plate φ_p(θ) [0-aligned]')
plt.plot(theta_deg_array_loss, phi_free_p_ref, '--', label='Free φ_p(θ) [0-aligned]')
plt.xlabel('Incident angle θ (deg)')
plt.ylabel('Phase (rad)')
plt.title(f'Angle-domain Phase (used in loss) 0–{theta_max_deg_for_loss:.0f}°')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

E_struct_after = run_simulation_from_E(E_after_gap, n_list_opt, d_list_opt, pol='s')
E_struct_ref   = angular_spectrum_1d(E_struct_after, wavelength, z_target, dx)
phase_struct_all = np.unwrap(np.angle(E_struct_ref))

x_mm = x * 1e3
c = Nx // 2
plt.figure(figsize=(8,4))
plt.plot(x_mm, phase_ref_eval, label="Free Space (centered)")
plt.plot(x_mm, phase_struct_all - (phase_struct_all[c] if CENTER_ALIGN_PHASE else 0.0),
         label="Optimized Plate (centered)")
plt.xlabel("x (mm)")
plt.ylabel("Phase (rad)")
plt.title("Phase Profile at z_target (full)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

ga_instance.plot_fitness()
