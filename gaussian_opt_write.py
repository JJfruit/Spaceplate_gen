import os, json, time, uuid, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from tmm_core_test import coh_tmm
import pygad
#Genetic Algorithm (GA)로 다층 박막 구조를 최적화하여
# 특정 빔 특성을 얻는 시뮬레이션 코드입니다
# ----------------------------
# 0) 실행 로깅/저장 유틸
# ----------------------------
LOG_ROOT = Path("ga_runs")

def make_run_paths(z_R):
    zR_tag = f"zR_{int(round(z_R*1e6))}um"
    zR_dir = LOG_ROOT / zR_tag
    zR_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + "-" + uuid.uuid4().hex[:6]
    run_dir = zR_dir / run_id
    run_dir.mkdir(exist_ok=True)
    return zR_dir, run_dir, run_id

class RunLogger:
    def __init__(self, z_R, note="", meta=None):
        self.z_R = z_R
        self.zR_dir, self.run_dir, self.run_id = make_run_paths(z_R)
        self.note = note
        self.meta = meta or {}
        self.fitness_rows = []  # (gen, best, mean, worst)
        self.t0 = time.time()

        # 미리 meta.json 생성
        meta_obj = {
            "run_id": self.run_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "z_R_um": float(z_R*1e6),
            "note": note,
            **self.meta
        }
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    def on_generation(self, ga):
        # 세대별 best/mean/worst 수집
        gen = ga.generations_completed
        pop_fit = getattr(ga, "last_generation_fitness", None)
        if pop_fit is None:
            # 폴백: 비용이 들 수 있음
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
        # per-run 상세
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        # zR별 요약 CSV에 한 줄 추가
        summary_csv = self.zR_dir / "runs_summary.csv"
        header = ["run_id","timestamp","z_R_um","compression_ratio_target","R_best",
                  "total_thickness_um","T_rms_best","phase_mse_space","phase_mse_angle",
                  "gen_at_best","gens_total","minutes_elapsed","note"]
        file_exists = summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow([
                self.run_id,
                time.strftime('%Y-%m-%d %H:%M:%S'),
                metrics_dict.get("z_R_um"),
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

# ----------------------------
# 1) 가우시안 빔 생성 함수
# ----------------------------
def Gaussian_Beam(A_1, z_0, lam_vac, X, z):
    """
    plate 면을 z=0으로 두고, z<0이면 '초점까지 남은 거리'를 의미합니다.
    """
    k = 2 * np.pi / lam_vac  # 파수
    W_0 = np.sqrt(lam_vac * z_0 / np.pi)      # 빔 최소 반경 (waist)
    W_z = W_0 * np.sqrt(1 + (z / z_0) ** 2)   # 빔 반경 변화

    # z=0에서 곡률 반경 R_z -> ∞ (곡률 위상항 = 1) 처리
    R_z = np.where(np.isclose(z, 0.0), np.inf, z * (1 + (z_0 / z) ** 2))
    Zeta_z = np.arctan(z / z_0)               # Gouy 위상

    # 곡률 위상항
    phase_curv = np.exp(-1j * k * (X**2 / (2 * R_z)))
    phase_curv = np.where(np.isfinite(R_z), phase_curv, 1.0)

    G_amp = (A_1 / (1j * z_0)) * (W_0 / W_z) * np.exp(-X**2 / W_z**2) \
            * np.exp(-1j * k * z) * phase_curv * np.exp(1j * Zeta_z)
    return G_amp

# ----------------------------
# 2) 파라미터
# ----------------------------
wavelength = 1550e-9
k0 = 2 * np.pi / wavelength

Nx = 128   # 샘플 수 (늘릴수록 해상도↑, 계산량도↑)

# ---- 사용자 지정: 자유공간 초점까지 거리와 비교면(d_eff)
z_focus_fs = 200e-6     # plate(구조체 입구)에서 자유공간 초점까지의 거리 (> d_eff)
d_eff       = 50e-6     # 포커스 "이전" 비교면 (자유공간 d_eff 전파면)

# 가우시안 파라미터
z_R = 85e-6
W0 = np.sqrt(wavelength * z_R / np.pi)   # 빔 최소 반경 (waist)

# x 범위: W0의 몇 배까지 커버할지 설정
x_range_factor = 20
x_max = x_range_factor * W0

x = np.linspace(-x_max, x_max, Nx)
dx = x[1] - x[0]

print(f"[Info] Beam waist W0 = {W0*1e6:.2f} µm, "
      f"x range = ±{x_max*1e6:.2f} µm, dx = {dx*1e9:.2f} nm")

assert d_eff < z_focus_fs, "d_eff는 포커스 이전이어야 합니다."

# GA/제약 관련
intensity_threshold_ratio = 0.5   # 전송 하한
phase_mse_threshold = 1e-3
angle_mse_threshold = 1e-1
weight_intensity = 0.1
compression_ratio = 5

# --- 위상 손실에 사용할 intensity 마스크 비율(레퍼런스 기준) ---
intensity_mask_ratio = 0.02       # 1% 권장

# === 추가: 손실 가중치 (공간 위상 vs 각도 위상) ===
W_SPACE = 0.5     # 공간(마스크) 위상 MSE 가중치
W_ANGLE = 4.0     # 각도 위상 MSE 가중치

# === 추가: 각도 위상 비교 범위/샘플 수 ===
theta_max_deg_for_loss = 10.0
N_theta_loss = 201  # 0~theta_max 범위에서 균등 샘플

# ----------------------------
# 3) 굴절률 스택
# ----------------------------
def alternating_n_list(N):
    n1, n2 = 3.6211 + 1.26e-7j, 1.4596 + 0.000011139j
    return [n1 if i % 2 == 0 else n2 for i in range(N)]

# ----------------------------
# 4) ASM 전파 (1D)
# ----------------------------
def angular_spectrum_1d(E_in, wavelength, distance, dx, n_medium=1.0):
    fx = np.fft.fftfreq(len(E_in), d=dx)      # [1/m]
    kx = 2 * np.pi * fx                       # [rad/m]
    k0 = 2 * np.pi / wavelength
    kz = np.sqrt((n_medium * k0)**2 - kx**2 + 0j)
    H = np.exp(1j * kz * distance)
    E_fft = fft(E_in)
    E_fft_prop = E_fft * H
    return ifft(E_fft_prop)

# ----------------------------
# 5) 레퍼런스(자유공간 d_eff 전파면)
# ----------------------------
def calculate_ref_output():
    E_in = Gaussian_Beam(A_1=0.1, z_0=z_R, lam_vac=wavelength,
                         X=x, z=-z_focus_fs)
    E_ref = angular_spectrum_1d(E_in, wavelength, d_eff, dx, n_medium=1.0)
    return E_in, E_ref

E_in, E_ref_field = calculate_ref_output()
phase_ref = np.unwrap(np.angle(E_ref_field))
I_ref = np.abs(E_ref_field)**2
I_ref_max = np.max(I_ref)
mask = I_ref >= (intensity_mask_ratio * I_ref_max)
if not np.any(mask):
    mask = np.zeros_like(I_ref, dtype=bool)
    mask[len(I_ref)//2] = True

# ----------------------------
# 6) 구조체 통과 후 출구장 (각도 의존 t를 스펙트럼에 곱함)
# ----------------------------
def run_simulation_from_E(E_input, n_list, d_list, pol='s'):
    fx = np.fft.fftfreq(Nx, d=dx)
    kx = 2 * np.pi * fx
    theta_kx = np.arcsin(np.clip(kx / k0, -1, 1))
    t_coef_array = np.zeros(Nx, dtype=complex)

    for i, theta in enumerate(theta_kx):
        try:
            tmm_result = coh_tmm(pol, n_list, d_list, theta, wavelength*1e9)
            t_coef_array[i] = tmm_result['t']
        except Exception:
            t_coef_array[i] = 0.0

    E_fft = fft(E_input)
    E_fft *= t_coef_array
    return ifft(E_fft)

# ----------------------------
# 7) 각도 평균 투과율
# ----------------------------
def compute_transmission(n_list, d_list, pol='s',
                             angle_range=np.linspace(-10, 10, 64), wavelength=1550):
    T_array = []
    for theta_deg in angle_range:
        theta_rad = np.deg2rad(theta_deg)
        try:
            tmm_result = coh_tmm(pol, n_list, d_list, theta_rad, wavelength)
            T_array.append(tmm_result['T'])
        except Exception:
            T_array.append(0.0)
    return np.mean(np.array(T_array))

# === 추가: 각도 위상 계산 유틸 ===
def transmission_phase_vs_theta(n_list, d_list, pol, theta_deg_array, lam_vac_nm):
    """주어진 스택의 투과 위상 φ(θ) [rad]를 θ 배열에 대해 계산(unwrap 포함)."""
    phases = np.zeros_like(theta_deg_array, dtype=float)
    for i, th in enumerate(theta_deg_array):
        try:
            th_rad = np.deg2rad(th)
            res = coh_tmm(pol, n_list, d_list, th_rad, lam_vac_nm)
            phases[i] = np.angle(res['t'])
        except Exception:
            phases[i] = np.nan
    return np.unwrap(phases)

# === 추가: 각도 위상 레퍼런스(자유공간 d_eff) 미리 계산 ===
theta_deg_array_loss = np.linspace(0.0, theta_max_deg_for_loss, N_theta_loss)
n_list_free = [1.0, 1.0, 1.0]
d_list_free = [np.inf, d_eff*1e9, np.inf]

phi_free_s_ref = transmission_phase_vs_theta(n_list_free, d_list_free, 's',
                                             theta_deg_array_loss, wavelength*1e9)
phi_free_p_ref = transmission_phase_vs_theta(n_list_free, d_list_free, 'p',
                                             theta_deg_array_loss, wavelength*1e9)
# θ=0 기준 상수위상 제거(각도 얼라인)
phi_free_s_ref -= phi_free_s_ref[0]
phi_free_p_ref -= phi_free_p_ref[0]

# ----------------------------
# (추가) 위상 차이 유틸: 2π 최단 차이
# ----------------------------
def angdiff(phi_ref, phi_pred):
    """
    위상 차이를 [-π, +π] 범위의 '최단' 차이로 반환.
    np.angle(exp(1j*Δφ)) = wrap_to_pi(Δφ)
    """
    return np.angle(np.exp(1j*(phi_ref - phi_pred)))

# ----------------------------
# 8) 피트니스 함수
# ----------------------------
CENTER_ALIGN_PHASE = True

def fitness_func(ga_instance, solution, solution_idx):
    d_solution = list(solution)
    d_list = [np.inf] + d_solution + [np.inf]
    n_list = [1.0] + alternating_n_list(len(d_solution)) + [1.0]

    total_thickness_m = sum(d_solution) * 1e-9
    R = d_eff / total_thickness_m
    if R < compression_ratio:
        return -1e5

    # --- 투과율 제약 ---
    T_rms_avg = 0.5 * (
        compute_transmission(n_list, d_list, 's') +
        compute_transmission(n_list, d_list, 'p')
    )
    if T_rms_avg < intensity_threshold_ratio:
        return -1e4

    # --- 공간 도메인 손실 (★수정: 2π 최단 위상 차이 사용) ---
    E_s = run_simulation_from_E(E_in, n_list, d_list, pol='s')
    E_p = run_simulation_from_E(E_in, n_list, d_list, pol='p')

    # unwrap 대신 원래 위상 사용
    phase_s = np.angle(E_s)
    phase_p = np.angle(E_p)
    phase_r = np.angle(E_ref_field)  # 레퍼런스도 angle로

    if CENTER_ALIGN_PHASE:
        c = Nx // 2
        phase_r = phase_r - phase_r[c]
        phase_s = phase_s - phase_s[c]
        phase_p = phase_p - phase_p[c]

    # 2π 주기 최단 차이로 마스크 구간 MSE
    diff_s_space = angdiff(phase_r, phase_s)
    diff_p_space = angdiff(phase_r, phase_p)
    loss_phase_space = 0.5 * (
        np.mean(diff_s_space[mask]**2) +
        np.mean(diff_p_space[mask]**2)
    )

    # === 각도 도메인 손실 (★수정: 2π 최단 위상 차이 사용) ===
    phi_plate_s = transmission_phase_vs_theta(n_list, d_list, 's',
                                              theta_deg_array_loss, wavelength*1e9)
    phi_plate_p = transmission_phase_vs_theta(n_list, d_list, 'p',
                                              theta_deg_array_loss, wavelength*1e9)
    # 기존 정렬 유지
    phi_plate_s -= phi_plate_s[0]
    phi_plate_p -= phi_plate_p[0]

    diff_s_theta = angdiff(phi_free_s_ref, phi_plate_s)
    diff_p_theta = angdiff(phi_free_p_ref, phi_plate_p)

    loss_phase_theta = 0.5 * (
        np.nanmean(diff_s_theta**2) +
        np.nanmean(diff_p_theta**2)
    )
    
    if loss_phase_theta > angle_mse_threshold:
        return -(1e2*loss_phase_theta)

    intensity_error = max(0.0, intensity_threshold_ratio - T_rms_avg)

    # --- 총 손실 (GA는 최대화 이므로 음수 반환) ---
    total_loss =  (W_ANGLE * loss_phase_theta) \
                 + weight_intensity * intensity_error
    return -total_loss

# ----------------------------
# 9) GA 콜백 (기존 + 로거 호출)
# ----------------------------
_last_fit = None
# 로거 생성 (여기에 실험 주석/메타를 자유롭게 넣으세요)
logger = RunLogger(
    z_R=z_R,
    note="baseline plate search",
    meta={
        "wavelength_nm": float(wavelength*1e9),
        "z_focus_fs_um": float(z_focus_fs*1e6),
        "d_eff_um": float(d_eff*1e6),
        "intensity_threshold_ratio": intensity_threshold_ratio,
        "phase_mse_threshold": phase_mse_threshold,
        "angle_mse_threshold": angle_mse_threshold,
        "W_SPACE": W_SPACE,
        "W_ANGLE": W_ANGLE,
        "theta_max_deg_for_loss": theta_max_deg_for_loss,
        "N_theta_loss": N_theta_loss,
        "Nx": Nx,
        "x_range_factor": x_range_factor,
        "compression_ratio_target": compression_ratio
    }
)

def on_generation(ga_instance):
    global _last_fit
    # 기존 출력
    current_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )[1]
    delta = 0 if _last_fit is None else current_fitness - _last_fit
    print(f"[Gen {ga_instance.generations_completed:02d}] "
          f"Best Fit: {current_fitness:.6f}  Δ:{delta:+.6f}")
    _last_fit = current_fitness

    # 로거에 기록
    logger.on_generation(ga_instance)

# ----------------------------
# 10) GA 실행
# ----------------------------
ga_instance = pygad.GA(
    num_generations=500,
    sol_per_pop=500,
    num_parents_mating=250,
    num_genes=25,
    gene_space={'low': 10, 'high': 1200},   # [nm]
    fitness_func=fitness_func,
    mutation_type="random",
    crossover_type="single_point",
    mutation_percent_genes=20,
    keep_elitism=3,
    on_generation=on_generation
#    random_seed=int(os.getenv("GA_SEED","0")) or None
)

ga_instance.run()

# 세대별 로그 파일로 저장
logger.flush_fitness_csv()

# ----------------------------
# 11) 결과 출력 및 평가 (+ 저장)
# ----------------------------
best_solution, best_fitness, _ = ga_instance.best_solution()
print("Best d_list (nm):", ', '.join(map(str, best_solution)))
total_thickness_m = sum(best_solution) * 1e-9
R_best = d_eff / total_thickness_m
print(f"Compression ratio R: {R_best:.2f}")

n_list_opt = [1.0] + alternating_n_list(len(best_solution)) + [1.0]
d_list_opt = [np.inf] + list(best_solution) + [np.inf]

E_s_best = run_simulation_from_E(E_in, n_list_opt, d_list_opt, pol='s')
E_p_best = run_simulation_from_E(E_in, n_list_opt, d_list_opt, pol='p')

phase_s_best = np.unwrap(np.angle(E_s_best))
phase_p_best = np.unwrap(np.angle(E_p_best))
c = Nx // 2
phase_r_eval = phase_ref - phase_ref[c]
phase_s_eval = phase_s_best - phase_s_best[c]
phase_p_eval = phase_p_best - phase_p_best[c]

phase_mse_best = 0.5 * (
    np.mean((phase_r_eval[mask] - phase_s_eval[mask])**2) +
    np.mean((phase_r_eval[mask] - phase_p_eval[mask])**2)
)
print(f"Best Phase MSE (avg s/p, masked): {phase_mse_best:.6e}")

T_rms_best = 0.5 * (
    compute_transmission(n_list_opt, d_list_opt, 's') +
    compute_transmission(n_list_opt, d_list_opt, 'p')
)
print(f"Best Intensity RMS (avg s/p): {T_rms_best:.4f}")

# === 추가: 각도 위상 MSE (평가 출력) ===
theta_deg_array_eval = theta_deg_array_loss  # 동일 범위로 평가
phi_plate_s_best = transmission_phase_vs_theta(n_list_opt, d_list_opt, 's',
                                               theta_deg_array_eval, wavelength*1e9)
phi_plate_p_best = transmission_phase_vs_theta(n_list_opt, d_list_opt, 'p',
                                               theta_deg_array_eval, wavelength*1e9)
phi_plate_s_best -= phi_plate_s_best[0]
phi_plate_p_best -= phi_plate_p_best[0]

theta_phase_mse_best = 0.5 * (
    np.nanmean((phi_free_s_ref - phi_plate_s_best)**2) +
    np.nanmean((phi_free_p_ref - phi_plate_p_best)**2)
)
print(f"Best Phase MSE (angle, 0–{theta_max_deg_for_loss:.0f}°, avg s/p): {theta_phase_mse_best:.6e}")
sum_nm = float(np.sum(best_solution))  # 총 두께 [nm]
print(f"Total stack thickness: {sum_nm:.2f} nm ({total_thickness_m*1e6:.3f} µm)")

# ---- (저장) best d_list / metrics / summary ----
logger.save_best_d_list(best_solution)

# gen@best 계산 (세대별 best fitness 중 최대가 처음 등장한 세대)
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
    "z_R_um": float(z_R*1e6),
    "compression_ratio_target": compression_ratio,
    "R_best": float(R_best),
    "total_thickness_um": float(total_thickness_m*1e6),
    "T_rms_best": float(T_rms_best),
    "phase_mse_space": float(phase_mse_best),
    "phase_mse_angle": float(theta_phase_mse_best),
    "gen_at_best": gen_at_best,
    "gens_total": gens_total,
    "minutes_elapsed": float(logger.elapsed_minutes()),
     "best_fitness": float(best_fitness)
}
logger.save_metrics(metrics)

print(f"[Saved] Run folder: {logger.run_dir}")
print(f"[Saved] Summary appended to: {logger.zR_dir / 'runs_summary.csv'}")

# ----------------------------
# 12) 위상: Plate vs Free Space(d_eff)
# ----------------------------
free_space_thickness_nm = d_eff * 1e9
n_list_free = [1.0, 1.0, 1.0]
d_list_free = [np.inf, free_space_thickness_nm, np.inf]

theta_deg_array = np.linspace(-90, 90, 1000)
phase_opt = np.zeros_like(theta_deg_array)
phase_free = np.zeros_like(theta_deg_array)

for i, theta_deg in enumerate(theta_deg_array):
    theta_rad = np.deg2rad(theta_deg)
    try:
        phase_opt[i] = np.angle(
            coh_tmm('s', n_list_opt, d_list_opt,
                    theta_rad, wavelength*1e9)['t']
        )
        phase_free[i] = np.angle(
            coh_tmm('s', n_list_free, d_list_free,
                    theta_rad, wavelength*1e9)['t']
        )
    except Exception:
        phase_opt[i] = phase_free[i] = np.nan

phase_opt = np.unwrap(phase_opt) - np.max(np.unwrap(phase_opt))
phase_free = np.unwrap(phase_free) - np.max(np.unwrap(phase_free))

plt.figure(figsize=(8,5))
plt.plot(theta_deg_array, phase_opt, label='Optimized Plate')
plt.plot(theta_deg_array, phase_free, '--', label='Free Space (d_eff)')
plt.xlim(0, 20)
plt.ylim(-10, 0.5)
plt.xlabel('Incident angle (deg)')
plt.ylabel('Transmission phase (rad)')
plt.title('Phase: Plate vs Free Space (thickness = d_eff)')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

# ----------------------------
# 13) 공간 위상 프로파일 (마스크 구간만 표시)
# ----------------------------
x_mm = x * 1e3
phase_struct = np.unwrap(np.angle(E_s_best))
center_idx = Nx // 2
phase_ref_centered = phase_ref - phase_ref[center_idx]
phase_struct_centered = phase_struct - phase_struct[center_idx]

plt.figure(figsize=(8,4))
plt.plot(x_mm[mask], phase_ref_centered[mask],
         label="Free Space @ d_eff (masked)")
plt.plot(x_mm[mask], phase_struct_centered[mask],
         label="Optimized Plate (exit, masked)")
plt.xlabel("x (mm)")
plt.ylabel("Phase (rad)")
plt.title("Phase Profile (center-normalized, intensity-masked)")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

#ga_instance.plot_fitness()

