import numpy as np
from numpy import pi, inf
from tmm_core_test import coh_tmm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 굴절률 리스트 생성 함수 ---
def alternating_n_list(N):
    n1, n2 = 3.6211 + 1.26e-7j, 1.4596 + 0.000011139j
    n_list = [1]  # 시작: 공기
    for i in range(N):
        n_list.append(n1 if i % 2 == 0 else n2)
    n_list.append(1)  # 끝: 공기
    return n_list

# --- d_list는 그대로 사용 ---
# R=5.38 MSE=1.8e-2 RMS Int=0.0821
d_list=[inf,114.52950162565553, 512.0468231799825, 343.4104566554145, 590.0846260503014, 136.5139226608373, 782.6505306125765, 366.8504495594343, 55.52169453104234, 837.4774300428987, 631.6336938622098, 477.3667661366504, 633.7500136279922, 145.43672874293773, 178.605532583041, 988.3806924953149, 48.46483484356094, 310.57927519304843, 767.5335307578164, 211.14460855680653, 449.9003185070341, 187.3735098596578, 258.80965872169725, 464.18302277922714, 279.7075393787748, 198.3681090187898,inf]

# --- n_list 생성 ---
num_layers = len(d_list) - 2
n_list = alternating_n_list(num_layers)

# --- n_list2, d_list2는 그대로 유지 ---
n_list2 = [1, 2, 1]
d_list2 = [inf, 10000, inf]

# --- 가우시안 빔 함수 정의 ---
def Gaussian_Beam(A_1, z_0, lam_vac, X, z):
    k = 2 * np.pi / lam_vac
    W_0 = np.sqrt(lam_vac * z_0 / np.pi)
    W_z = W_0 * np.sqrt(1 + (z / z_0) ** 2)
    R_z = z * (1 + (z_0 / z) ** 2)
    Zeta_z = np.arctan(z / z_0)
    G_amp = (A_1 / (1j * z_0)) * (W_0 / W_z) * np.exp(-X**2 / W_z**2) * \
            np.exp(-1j * k * z) * np.exp(-1j * k * (X**2 / (2 * R_z))) * np.exp(1j * Zeta_z)
    return G_amp

# --- 파라미터 설정 ---
wavelength = 1550e-9
k = 2 * np.pi / wavelength
z_max = 0.0005 / 2
Nz = 10000

Nx = 4000
Lx = 0.5e-3
dx = Lx / Nx
x = np.linspace(-Lx/2, Lx/2, Nx)
z = np.linspace(-z_max, z_max, Nz)
z = np.where(z == 0, 1e-12, z)
X, Z = np.meshgrid(x, z)

z_0 = 40e-6
E0 = Gaussian_Beam(0.1, z_0, wavelength, x, z_max)

fx = np.fft.fftfreq(Nx, dx)
kx = 2 * np.pi * fx
k0 = 2 * np.pi / wavelength
theta = np.arcsin(np.clip(kx / k0, -1, 1))

tmm_t  = np.zeros_like(kx, dtype=complex)
tmm_t2 = np.zeros_like(kx, dtype=complex)
for i, th in enumerate(theta):
    if np.abs(kx[i]) <= k0:
        tmm_t[i]  = coh_tmm('s', n_list, d_list, th, 1550)['t']
        tmm_t2[i] = coh_tmm('p', n_list, d_list, th, 1550)['t']
    else:
        tmm_t[i]  = 0
        tmm_t2[i] = 0

# --- Free Space 전파 ---
E_prop = np.zeros((Nz, Nx), dtype=complex)
E_prop[0, :] = E0
for i in range(1, Nz):
    dz = z[i] - z[i - 1]
    kz = np.sqrt(k**2 - kx**2, dtype=complex)
    H = np.exp(1j * kz * dz)
    E_prop[i, :] = np.fft.ifft(np.fft.fft(E_prop[i - 1, :]) * H)

# --- Space Plate (s-pol) 전파 ---
E0_new  = np.fft.ifft(np.fft.fft(E0) * tmm_t)
E_prop2 = np.zeros((Nz, Nx), dtype=complex)
E_prop2[0, :] = E0_new
for i in range(1, Nz):
    dz = z[i] - z[i - 1]
    kz = np.sqrt(k**2 - kx**2, dtype=complex)
    H = np.exp(1j * kz * dz)
    E_prop2[i, :] = np.fft.ifft(np.fft.fft(E_prop2[i - 1, :]) * H) * np.exp(1j * np.arctan(z[i] / z_0))

# --- Space Plate (p-pol) 전파 ---
E0_new3 = np.fft.ifft(np.fft.fft(E0) * tmm_t2)
E_prop3 = np.zeros((Nz, Nx), dtype=complex)
E_prop3[0, :] = E0_new3
for i in range(1, Nz):
    dz = z[i] - z[i - 1]
    kz = np.sqrt(k**2 - kx**2, dtype=complex)
    H = np.exp(1j * kz * dz)
    E_prop3[i, :] = np.fft.ifft(np.fft.fft(E_prop3[i - 1, :]) * H) * np.exp(1j * np.arctan(z[i] / z_0))

# --- Focal 위치 계산 ---
z_max_idx_1 = np.argmax(np.abs(E_prop[:,  Nx//2])**2)
z_max_idx_2 = np.argmax(np.abs(E_prop2[:, Nx//2])**2)
z_max_idx_3 = np.argmax(np.abs(E_prop3[:, Nx//2])**2)
z_max_1, z_max_2, z_max_3 = z[z_max_idx_1], z[z_max_idx_2], z[z_max_idx_3]
d_total = sum(d for d in d_list if d != inf) * 1e-9


# --- offset adjust ---
z_off   = z_max_1          # meters
z_plot  = z - z_off        # 플로팅용 (표시용) z축
z1_plot = z_max_1 - z_off  # == 0
z2_plot = z_max_2 - z_off
z3_plot = z_max_3 - z_off

# --- 시각화 (필드 맵 & z-강도) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

im1 = axes[0,0].imshow(np.abs(E_prop.T)**2/np.max(np.abs(E_prop)**2),
                       extent=[z_plot[0], z_plot[-1], -Lx/2, Lx/2],
                       aspect='auto', cmap='inferno')
axes[0,0].axvline(z1_plot, color='gray', linestyle='--', label='Free Space')
axes[0,0].set_xlabel('Propagation distance z (um)')
axes[0,0].set_ylabel(' x (um)')
axes[0,0].set_title('Gaussian Beam w/o Spaceplate')
axes[0,0].ticklabel_format(axis='both', useOffset=False, style='sci', scilimits=(-6,-6))
axes[0,0].set_xlim(-200e-6,200e-6)
axes[0,0].set_ylim(-200e-6,200e-6)
fig.colorbar(im1, ax=axes[0,0], label='Intensity |E|^2')
axes[0,0].legend(fontsize=8)

intensity_x0 = np.abs(E_prop[:, Nx//2])**2
axes[1,0].plot(z_plot, intensity_x0/np.max(np.abs(E_prop)**2), color='blue')
axes[1,0].axvline(z1_plot, color='gray', linestyle='--', label='Free Space')
axes[1,0].set_xlabel('Propagation distance z (m)')
axes[1,0].set_ylabel('Intensity')
axes[1,0].set_title('Intensity(w/o SP)')
axes[1,0].ticklabel_format(axis='x', useOffset=False, style='sci', scilimits=(-6,-6))
axes[1,0].set_xlim(-200e-6,200e-6)
axes[1,0].legend(fontsize=8)

im2 = axes[0,1].imshow(np.abs(E_prop2.T)**2/np.max(np.abs(E_prop2)**2),
                       extent=[z_plot[0], z_plot[-1], -Lx/2, Lx/2],
                       aspect='auto', cmap='inferno')
axes[0,1].axvline(z1_plot, color='gray', linestyle='--', label='Free Space')
axes[0,1].axvline(z2_plot, color='red', linestyle='--', label='w/ Space Plate')
axes[0,1].set_xlabel('Propagation distance z (um)')
axes[0,1].set_ylabel(' x (um)')
axes[0,1].set_title('Gaussian Beam w/ Spaceplate')
axes[0,1].ticklabel_format(axis='both', useOffset=False, style='sci', scilimits=(-6,-6))
axes[0,1].set_xlim(-200e-6,200e-6)
axes[0,1].set_ylim(-200e-6,200e-6)
fig.colorbar(im2, ax=axes[0,1], label='Intensity |E|^2')
axes[0,1].legend(fontsize=8)

intensity_x2 = np.abs(E_prop2[:, Nx//2])**2
intensity_x3 = np.abs(E_prop3[:, Nx//2])**2
axes[1,1].plot(z_plot, intensity_x2/np.max(np.abs(E_prop2)**2), color='blue', label='s-pol')
axes[1,1].plot(z_plot, intensity_x3/np.max(np.abs(E_prop3)**2), color='red', linestyle='--', label='p-pol')
axes[1,1].axvline(z1_plot, color='gray', linestyle='--', label='Free Space')
axes[1,1].axvline(z2_plot, color='blue', linestyle='solid')
axes[1,1].axvline(z3_plot, color='red', linestyle='--')
axes[1,1].set_xlabel('Propagation distance z (m)')
axes[1,1].set_ylabel('Intensity')
axes[1,1].set_title('Intensity(w/ SP)')
axes[1,1].ticklabel_format(axis='x', useOffset=False, style='sci', scilimits=(-6,-6))
axes[1,1].set_xlim(-200e-6,200e-6)
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# 여기서부터 정규화 기반 피팅
#   - x: Gaussian
#   - z: Rayleigh(shifted)  I(z)=I0 / (1 + ((z - zc)/zR)^2) + C
# =========================

# ---------- 피팅 유틸 ----------
def fit_with_model(xv, yv, model, p0, bounds=None):
    if bounds is None:
        popt, pcov = curve_fit(model, xv, yv, p0=p0, maxfev=20000)
    else:
        popt, pcov = curve_fit(model, xv, yv, p0=p0, bounds=bounds, maxfev=20000)
    residuals = yv - model(xv, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yv - np.mean(yv))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return popt, pcov, r2

# ---------- 모델들 ----------
def gaussian(xv, A, mu, sigma, C):
    return A * np.exp(-((xv - mu) ** 2) / (2.0 * sigma ** 2)) + C

def rayleigh_shifted(zv, I0, zc, zR, C):
    # 축방향 가우시안 빔 강도 (중심 zc 포함)
    return I0 / (1.0 + ((zv - zc) / zR) ** 2) + C

def FWHM_from_sigma(sigma):  # x-가우시안
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

def FWHM_from_zR(zR):        # z-Rayleigh
    return 2.0 * zR

# ---------- 초기 추정치 ----------
def init_guess_gaussian(xv, yv):
    A0 = (np.max(yv) - np.min(yv))
    mu0 = xv[np.argmax(yv)]
    sigma0 = (xv[-1] - xv[0]) / 10.0
    C0 = np.min(yv)
    return [A0, mu0, sigma0, C0]

def init_guess_rayleigh_shifted(zv, yv, zc0):
    I0 = np.max(yv)
    zR = (zv[-1] - zv[0]) / 10.0
    C0 = np.min(yv)
    return [I0, zc0, zR, C0]

# 선택: z-피팅 윈도우 (um)
win_um = 200.0   # 필요 없으면 0으로
win = win_um * 1e-6

def slice_window(zv, yv, zc, w):
    if w is None or w <= 0:
        return zv, yv
    m = np.abs(zv - zc) <= w
    return zv[m], yv[m]

# ---------- 프로필 추출 ----------
# z-프로필 (중심 x)
I_z_free = np.abs(E_prop[:,  Nx//2])**2
I_z_s    = np.abs(E_prop2[:, Nx//2])**2
I_z_p    = np.abs(E_prop3[:, Nx//2])**2

# x-프로필 (각 케이스 초점 z에서)
I_x_free = np.abs(E_prop [z_max_idx_1, :])**2
I_x_s    = np.abs(E_prop2[z_max_idx_2, :])**2
I_x_p    = np.abs(E_prop3[z_max_idx_3, :])**2

# ---------- 정규화 ----------
I_x_free /= np.max(I_x_free);  I_x_s /= np.max(I_x_s);  I_x_p /= np.max(I_x_p)
I_z_free /= np.max(I_z_free);  I_z_s /= np.max(I_z_s);  I_z_p /= np.max(I_z_p)

# ---------- 피팅: x (Gaussian) ----------
pg_free, _, r2_g_free = fit_with_model(x, I_x_free, gaussian, init_guess_gaussian(x, I_x_free))
pg_s,    _, r2_g_s    = fit_with_model(x, I_x_s,    gaussian, init_guess_gaussian(x, I_x_s))
pg_p,    _, r2_g_p    = fit_with_model(x, I_x_p,    gaussian, init_guess_gaussian(x, I_x_p))

FWHM_x_free = FWHM_from_sigma(pg_free[2])
FWHM_x_s    = FWHM_from_sigma(pg_s[2])
FWHM_x_p    = FWHM_from_sigma(pg_p[2])

# ---------- 피팅: z (Rayleigh with center) ----------
# 중심 초기값은 각각의 초점 위치 사용
z0_free, z0_s, z0_p = z_max_1, z_max_2, z_max_3

# Free
z_fit_free, I_free_fit = slice_window(z, I_z_free, z0_free, win)
p0_r_free = init_guess_rayleigh_shifted(z_fit_free, I_free_fit, z0_free)
bounds_free = ([0.0, z0_free - 5*win, 0.0, -0.1],
               [2.0, z0_free + 5*win, (z[-1]-z[0]), 0.5])
pr_free, _, r2_r_free = fit_with_model(z_fit_free, I_free_fit, rayleigh_shifted, p0_r_free, bounds=bounds_free)
FWHM_z_free = FWHM_from_zR(pr_free[2])

# s-pol
z_fit_s, I_s_fit = slice_window(z, I_z_s, z0_s, win)
p0_r_s = init_guess_rayleigh_shifted(z_fit_s, I_s_fit, z0_s)
bounds_s = ([0.0, z0_s - 5*win, 0.0, -0.1],
            [2.0, z0_s + 5*win, (z[-1]-z[0]), 0.5])
pr_s, _, r2_r_s = fit_with_model(z_fit_s, I_s_fit, rayleigh_shifted, p0_r_s, bounds=bounds_s)
FWHM_z_s = FWHM_from_zR(pr_s[2])

# p-pol
z_fit_p, I_p_fit = slice_window(z, I_z_p, z0_p, win)
p0_r_p = init_guess_rayleigh_shifted(z_fit_p, I_p_fit, z0_p)
bounds_p = ([0.0, z0_p - 5*win, 0.0, -0.1],
            [2.0, z0_p + 5*win, (z[-1]-z[0]), 0.5])
pr_p, _, r2_r_p = fit_with_model(z_fit_p, I_p_fit, rayleigh_shifted, p0_r_p, bounds=bounds_p)
FWHM_z_p = FWHM_from_zR(pr_p[2])

# ---------- 결과 출력 ----------
print("=== Gaussian fit (x-profile @ focus, normalized) ===")
print(f"[Free]  A={pg_free[0]:.3f}, mu={pg_free[1]:.3e} m, sigma={pg_free[2]:.3e} m, C={pg_free[3]:.3f},  R^2={r2_g_free:.5f}, FWHM={FWHM_x_free*1e6:.3f} um")
print(f"[s-pol] A={pg_s[0]:.3f}, mu={pg_s[1]:.3e} m, sigma={pg_s[2]:.3e} m, C={pg_s[3]:.3f},  R^2={r2_g_s:.5f},    FWHM={FWHM_x_s*1e6:.3f} um")
print(f"[p-pol] A={pg_p[0]:.3f}, mu={pg_p[1]:.3e} m, sigma={pg_p[2]:.3e} m, C={pg_p[3]:.3f},  R^2={r2_g_p:.5f},    FWHM={FWHM_x_p*1e6:.3f} um")

print("\n=== Rayleigh (shifted center) fit (z-profile @ x=0, normalized) ===")
print(f"[Free]  I0={pr_free[0]:.3f}, zc={pr_free[1]*1e6:.3f} um, zR={pr_free[2]*1e6:.3f} um, C={pr_free[3]:.3f}, R^2={r2_r_free:.5f}, FWHM_z={FWHM_z_free*1e6:.3f} um")
print(f"[s-pol] I0={pr_s[0]:.3f}, zc={pr_s[1]*1e6:.3f} um, zR={pr_s[2]*1e6:.3f} um, C={pr_s[3]:.3f}, R^2={r2_r_s:.5f},    FWHM_z={FWHM_z_s*1e6:.3f} um")
print(f"[p-pol] I0={pr_p[0]:.3f}, zc={pr_p[1]*1e6:.3f} um, zR={pr_p[2]*1e6:.3f} um, C={pr_p[3]:.3f}, R^2={r2_r_p:.5f},    FWHM_z={FWHM_z_p*1e6:.3f} um")

# ---------- 피팅 곡선 시각화 (정규화 데이터, 새 Figure) ----------
fit_x_free = gaussian(x, *pg_free)
fit_x_s    = gaussian(x, *pg_s)
fit_x_p    = gaussian(x, *pg_p)

fit_z_free = rayleigh_shifted(z, *pr_free)
fit_z_s    = rayleigh_shifted(z, *pr_s)
fit_z_p    = rayleigh_shifted(z, *pr_p)

fig_fit, axes_fit = plt.subplots(2, 3, figsize=(13, 7))

# x-profile @ focus
axes_fit[0,0].plot(x*1e6, I_x_free, lw=1, label='Data')
axes_fit[0,0].plot(x*1e6, fit_x_free, lw=1.5, linestyle='--', label='Gaussian fit')
axes_fit[0,0].set_title('x-profile @ focus (Freespace)')
axes_fit[0,0].set_xlabel('x (um)'); axes_fit[0,0].set_ylabel('Normalized Intensity'); axes_fit[0,0].legend()
axes_fit[0,0].set_xlim(-15,15)

axes_fit[0,1].plot(x*1e6, I_x_s, lw=1, label='Data')
axes_fit[0,1].plot(x*1e6, fit_x_s, lw=1.5, linestyle='--', label='Gaussian fit')
axes_fit[0,1].set_title('x-profile @ focus (Spaceplate, s-pol)')
axes_fit[0,1].set_xlabel('x (um)'); axes_fit[0,1].set_ylabel('Normalized Intensity'); axes_fit[0,1].legend()
axes_fit[0,1].set_xlim(-15,15)

axes_fit[0,2].plot(x*1e6, I_x_p, lw=1, label='Data')
axes_fit[0,2].plot(x*1e6, fit_x_p, lw=1.5, linestyle='--', label='Gaussian fit')
axes_fit[0,2].set_title('x-profile @ focus (Spaceplate, p-pol)')
axes_fit[0,2].set_xlabel('x (um)'); axes_fit[0,2].set_ylabel('Normalized Intensity'); axes_fit[0,2].legend()
axes_fit[0,2].set_xlim(-15,15)

# z-profile @ center x=0  (Rayleigh shifted)
axes_fit[1,0].plot(z_plot*1e6, I_z_free, lw=1, label='Data')
axes_fit[1,0].plot(z_plot*1e6, fit_z_free, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,0].axvline(z1_plot*1e6, color='gray', linestyle=':', label='z_focus (free)')
axes_fit[1,0].set_title('z-profile (Freespace)')
axes_fit[1,0].set_xlabel('z (um)'); axes_fit[1,0].set_ylabel('Normalized Intensity'); axes_fit[1,0].legend()
axes_fit[1,0].set_xlim(-200,200)

axes_fit[1,1].plot(z_plot*1e6, I_z_s, lw=1, label='Data')
axes_fit[1,1].plot(z_plot*1e6, fit_z_s, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,1].axvline(z2_plot*1e6, color='gray', linestyle=':', label='z_focus (s)')
axes_fit[1,1].set_title('z-profile (Spaceplate, s-pol)')
axes_fit[1,1].set_xlabel('z (um)'); axes_fit[1,1].set_ylabel('Normalized Intensity'); axes_fit[1,1].legend()
axes_fit[1,1].set_xlim(-200,200)

axes_fit[1,2].plot(z_plot*1e6, I_z_p, lw=1, label='Data')
axes_fit[1,2].plot(z_plot*1e6, fit_z_p, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,2].axvline(z3_plot*1e6, color='gray', linestyle=':', label='z_focus (p)')
axes_fit[1,2].set_title('z-profile (Spaceplate, p-pol)')
axes_fit[1,2].set_xlabel('z (um)'); axes_fit[1,2].set_ylabel('Normalized Intensity'); axes_fit[1,2].legend()
axes_fit[1,2].set_xlim(-200,200)

plt.tight_layout()
plt.show()
