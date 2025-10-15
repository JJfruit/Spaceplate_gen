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

#후보1 (R)
#d_list=[inf,103.76026496817717, 1039.511569681379, 337.32809502872993, 137.26590286279475, 298.7480510423004, 67.32628895600038, 561.6397423328146, 285.22031425818426, 778.2482667198801, 1051.461634340994, 28.650003285361272, 21.550720103742087, 254.39286416888078, 887.4826901053451, 228.8036832922154, 1129.6569998549198, 550.6291051491514, 870.1087050987517, 329.957902780657, 19.459252126889623, 58.99627507177938, 407.4796163870822, 65.44076379250879, 161.44143021910804, 133.7250185223764,inf]
#후보1-2 (zR : 40/ R:5.08 / fit : -8.206 / I : 0.729 / MSE : 8.21e-3 / After : 382 / time : 0:26)
#d_list=[inf,998.6417504972609, 287.1820153581964, 61.15050066567897, 797.8229145296937, 640.5266236571915, 287.74703389981437, 586.7025507919419, 468.35201676193697, 148.99017124331644, 365.5975112515602, 247.86901619235024, 45.16698618677775, 781.3385816241072, 1050.5828130313048, 279.5017399560469, 58.93244075126443, 65.01225926697552, 267.6278577394406, 103.95574851093308, 500.25544132573293, 108.75472076499302, 529.5843713557213, 104.42992275074083, 669.8765555845661, 394.5663284105967,inf]

#후보2-1 (zR : 65/ R:5.96 / fit : -1.086 / I : 0.356 / MSE : 1.086e-3 / After : 31 / time : 0:30)
#d_list=[inf,133.2732528272806, 361.62016263215634, 149.30690199169376, 302.1352564991626, 225.69102175851754, 229.05036649088487, 169.39185335517715, 728.933788133559, 281.44391176895846, 557.6117011550939, 557.3522063887651, 131.5777144582301, 1086.1061295808104, 108.52305191464063, 76.19498770475222, 656.6681547279162, 609.1893769922966, 485.99430211296317, 270.09695908467717, 130.6628721958527, 54.011931415716354, 753.7631715344934, 12.330961767049526, 283.8641528430208, 41.49175286114737,inf]

#후보3-1 (zR : 35/ R:5.8 / fit : -37.230 / I : 0.464 / MSE : 3.72e-3 / After : 160 / time : 0:15)
#d_list=[inf,332.09629292556326, 168.5522259662994, 602.4475534787115, 105.49370523952577, 34.45256906023593, 232.58493704214908, 922.976561345035, 227.99779496729653, 265.57847022542353, 704.1686832609511, 347.547297077915, 23.33173430156895, 136.47007399416424, 864.5877712797417, 162.9764911246659, 361.493668002317, 410.6234926166285, 159.87736641861093, 86.35395579531988, 308.51929477587527, 498.3573113401448, 327.12504668015765, 520.2032919468481, 565.8987567185675, 249.65147425728813,inf]

#테스트 (zR : 43.8252/ R:5.8 / fit : -12.230 / I : 0.502 / MSE : 3.72e-3 / After : 170 / time : 0:25)
#d_list=[inf, 149.8049807185412, 624.3318548877364, 226.15024834689214, 111.21812629186464, 164.1634135802422, 291.15894822082686, 637.7783993757416, 51.68125192655414, 97.11334409528035, 980.5648321678946, 595.5817249125572, 219.05318152423268, 609.4394297202223, 49.84538656088094, 49.92236075522745, 401.70285253227104, 268.4211374854179, 491.9790630588637, 127.0219480959777, 126.04444571067889, 444.41809102808764, 84.98850570761616, 130.19988182241164, 255.13518447235796, 817.7634930650659,inf]

#테스트 이거 대박 (zR : 43.8252/ R:5.12 / fit : -133.857 / I : 0.5086 / MSE : 2.24e-2 / After : 291 / time : 0:12)
#d_list=[inf,114.84295422733014, 306.8495500746253, 56.82127910039784, 662.4946281110156, 95.12338578112849, 988.5047334705555, 880.443002232476, 64.78295982828777, 116.25913136184907, 235.93559881741302, 138.248957784072, 774.7150723625751, 167.70621509662294, 820.1001265511082, 570.3592176982108, 172.99901858516168, 1006.6303706261211, 441.39775309592756, 484.6560141563344, 66.5935347020978, 213.4965545566498, 254.32076937041296, 863.223754020751, 144.8722122246229, 128.92885411341126,inf]

d_list=[inf,150.33334564912994,
    410.3715344004813,
    191.66536231994826,
    1139.2505295026765,
    127.80200423653824,
    480.0110168026548,
    595.509553528162,
    503.7920028807836,
    352.9184563859952,
    709.3388225345157,
    353.4605298475678,
    123.18324854354012,
    334.45842369994523,
    755.8205458217723,
    76.70530516169298,
    12.460127663687807,
    327.09596287880333,
    399.6685089795451,
    605.4458353793367,
    499.9566319057401,
    17.71420166217225,
    43.68088651781419,
    326.36014892356076,
    692.9618076390107,
    335.70042674850964,inf]

# --- n_list 생성 ---
num_layers = len(d_list) - 2
n_list = alternating_n_list(num_layers)


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

z_0 = 65e-6
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
axes[0,0].axvline(z1_plot, color='gray', linestyle='--', label='Focus (Freespace)')
axes[0,0].set_xlabel('Distance z (um)',fontsize=13)
axes[0,0].set_ylabel(' x (um)',fontsize=13)
axes[0,0].set_title('Gaussian Beam w/o Spaceplate' ,fontsize=15)
axes[0,0].ticklabel_format(axis='both', useOffset=False, style='sci', scilimits=(-6,-6))
axes[0,0].tick_params(axis='both',labelsize=13)
axes[0,0].set_xlim(-200e-6,200e-6)
axes[0,0].set_ylim(-200e-6,200e-6)
fig.colorbar(im1, ax=axes[0,0], label='Intensity |E|^2')
axes[0,0].legend(fontsize=9)

intensity_x0 = np.abs(E_prop[:, Nx//2])**2
axes[1,0].plot(z_plot, intensity_x0/np.max(np.abs(E_prop)**2), color='blue')
axes[1,0].axvline(z1_plot, color='gray', linestyle='--', label='Focus (Freespace)')
axes[1,0].set_xlabel('Distance z (m)',fontsize=13)
axes[1,0].set_ylabel('Intensity',fontsize=13)
axes[1,0].set_title('Intensity(w/o SP)',fontsize=15)
axes[1,0].ticklabel_format(axis='x', useOffset=False, style='sci', scilimits=(-6,-6))
axes[1,0].tick_params(axis='both',labelsize=13)
axes[1,0].set_xlim(-200e-6,200e-6)
axes[1,0].legend(fontsize=9,loc='upper right')

im2 = axes[0,1].imshow(np.abs(E_prop2.T)**2/np.max(np.abs(E_prop2)**2),
                       extent=[z_plot[0], z_plot[-1], -Lx/2, Lx/2],
                       aspect='auto', cmap='inferno')
axes[0,1].axvline(z1_plot, color='gray', linestyle='--', label='Focus (Freespace)')
axes[0,1].axvline(z2_plot, color='red', linestyle='--', label='Focus (Spaceplate)')
axes[0,1].set_xlabel('Distance z (um)',fontsize=13)
axes[0,1].set_ylabel(' x (um)',fontsize=13)
axes[0,1].set_title('Gaussian Beam w/ Spaceplate',fontsize=15)
axes[0,1].ticklabel_format(axis='both', useOffset=False, style='sci', scilimits=(-6,-6))
axes[0,1].tick_params(axis='both',labelsize=13)
axes[0,1].set_xlim(-200e-6,200e-6)
axes[0,1].set_ylim(-200e-6,200e-6)
fig.colorbar(im2, ax=axes[0,1], label='Intensity |E|^2')
axes[0,1].legend(fontsize=9)

intensity_x2 = np.abs(E_prop2[:, Nx//2])**2
intensity_x3 = np.abs(E_prop3[:, Nx//2])**2
axes[1,1].plot(z_plot, intensity_x2/np.max(np.abs(E_prop2)**2), color='blue', label='s-pol')
axes[1,1].plot(z_plot, intensity_x3/np.max(np.abs(E_prop3)**2), color='red', linestyle='--', label='p-pol')
axes[1,1].axvline(z1_plot, color='gray', linestyle='--', label='Focus (Freespace)')
axes[1,1].axvline(z2_plot, color='blue', linestyle='solid')
axes[1,1].axvline(z3_plot, color='red', linestyle='--')
axes[1,1].set_xlabel('Distance z (m)',fontsize=13)
axes[1,1].set_ylabel('Intensity',fontsize=13)
axes[1,1].set_title('Intensity(w/ SP)',fontsize=15)
axes[1,1].ticklabel_format(axis='x', useOffset=False, style='sci', scilimits=(-6,-6))
axes[1,1].tick_params(axis='both',labelsize=13)
axes[1,1].set_xlim(-200e-6,200e-6)
axes[1,1].legend(fontsize=9)

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
win_um = 150.0   # 필요 없으면 0으로
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

# ---------- 피팅: z (Rayleigh with center fixed to focus) ----------
# 중심은 각각의 초점 위치로 '고정'하여 평행이동 가정과 일치시키고,
# I0, zR, C만 피팅합니다. (pr_*는 [I0, zc_fixed, zR, C]로 재조립)

def init_guess_center_fixed(zv, yv):
    I0 = float(np.max(yv))
    zR = (zv[-1] - zv[0]) / 10.0
    C0 = float(np.min(yv))
    return [I0, zR, C0]

bounds_center_fixed = ([0.0, 0.0, -0.1],
                       [2.0, (z[-1]-z[0]), 0.5])

# 중심 초기값은 각각의 초점 위치 사용
z0_free, z0_s, z0_p = z_max_1, z_max_2, z_max_3

# Free (zc 고정 = z0_free)
z_fit_free, I_free_fit = slice_window(z, I_z_free, z0_free, win)
p0_free = init_guess_center_fixed(z_fit_free, I_free_fit)
model_free = lambda zv, I0, zR, C: rayleigh_shifted(zv, I0, z0_free, zR, C)
popt_free, _, r2_r_free = fit_with_model(z_fit_free, I_free_fit, model_free,
                                         p0_free, bounds=bounds_center_fixed)
pr_free = np.array([popt_free[0], z0_free, popt_free[1], popt_free[2]], dtype=float)
FWHM_z_free = FWHM_from_zR(pr_free[2])

# s-pol (zc 고정 = z0_s)
z_fit_s, I_s_fit = slice_window(z, I_z_s, z0_s, win)
p0_s = init_guess_center_fixed(z_fit_s, I_s_fit)
model_s = lambda zv, I0, zR, C: rayleigh_shifted(zv, I0, z0_s, zR, C)
popt_s, _, r2_r_s = fit_with_model(z_fit_s, I_s_fit, model_s,
                                   p0_s, bounds=bounds_center_fixed)
pr_s = np.array([popt_s[0], z0_s, popt_s[1], popt_s[2]], dtype=float)
FWHM_z_s = FWHM_from_zR(pr_s[2])

# p-pol (zc 고정 = z0_p)
z_fit_p, I_p_fit = slice_window(z, I_z_p, z0_p, win)
p0_p = init_guess_center_fixed(z_fit_p, I_p_fit)
model_p = lambda zv, I0, zR, C: rayleigh_shifted(zv, I0, z0_p, zR, C)
popt_p, _, r2_r_p = fit_with_model(z_fit_p, I_p_fit, model_p,
                                   p0_p, bounds=bounds_center_fixed)
pr_p = np.array([popt_p[0], z0_p, popt_p[1], popt_p[2]], dtype=float)
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
axes_fit[0,0].set_title('x-profile @ focus (Freespace)',fontsize=15)
axes_fit[0,0].set_xlabel('x (um)',fontsize=13); axes_fit[0,0].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[0,0].legend()
axes_fit[0,0].set_xlim(-15,15)

axes_fit[0,1].plot(x*1e6, I_x_s, lw=1, label='Data')
axes_fit[0,1].plot(x*1e6, fit_x_s, lw=1.5, linestyle='--', label='Gaussian fit')
axes_fit[0,1].set_title('x-profile @ focus (Spaceplate, s-pol)',fontsize=15)
axes_fit[0,1].set_xlabel('x (um)',fontsize=13); axes_fit[0,1].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[0,1].legend()
axes_fit[0,1].set_xlim(-15,15)

axes_fit[0,2].plot(x*1e6, I_x_p, lw=1, label='Data')
axes_fit[0,2].plot(x*1e6, fit_x_p, lw=1.5, linestyle='--', label='Gaussian fit')
axes_fit[0,2].set_title('x-profile @ focus (Spaceplate, p-pol)',fontsize=15)
axes_fit[0,2].set_xlabel('x (um)',fontsize=13); axes_fit[0,2].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[0,2].legend()
axes_fit[0,2].set_xlim(-15,15)

# z-profile @ center x=0  (Rayleigh shifted)
axes_fit[1,0].plot(z_plot*1e6, I_z_free, lw=1, label='Data')
axes_fit[1,0].plot(z_plot*1e6, fit_z_free, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,0].axvline(z1_plot*1e6, color='gray', linestyle=':', label='z_focus (free)')
axes_fit[1,0].set_title('z-profile (Freespace)',fontsize=15)
axes_fit[1,0].set_xlabel('z (um)',fontsize=13); axes_fit[1,0].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[1,0].legend()
axes_fit[1,0].set_xlim(-200,200)
axes_fit[1,0].legend(loc='upper right')

axes_fit[1,1].plot(z_plot*1e6, I_z_s, lw=1, label='Data')
axes_fit[1,1].plot(z_plot*1e6, fit_z_s, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,1].axvline(z2_plot*1e6, color='gray', linestyle=':', label='z_focus (s)')
axes_fit[1,1].set_title('z-profile (Spaceplate, s-pol)',fontsize=15)
axes_fit[1,1].set_xlabel('z (um)',fontsize=13); axes_fit[1,1].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[1,1].legend()
axes_fit[1,1].set_xlim(-200,200)

axes_fit[1,2].plot(z_plot*1e6, I_z_p, lw=1, label='Data')
axes_fit[1,2].plot(z_plot*1e6, fit_z_p, lw=1.5, linestyle='--', label='Rayleigh fit')
axes_fit[1,2].axvline(z3_plot*1e6, color='gray', linestyle=':', label='z_focus (p)')
axes_fit[1,2].set_title('z-profile (Spaceplate, p-pol)',fontsize=15)
axes_fit[1,2].set_xlabel('z (um)',fontsize=13); axes_fit[1,2].set_ylabel('Normalized Intensity',fontsize=13); axes_fit[1,2].legend()
axes_fit[1,2].set_xlim(-200,200)

plt.tight_layout()
plt.show()
