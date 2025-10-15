import numpy as np
from numpy import pi, inf
from tmm_core_test import coh_tmm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# ===============================
# 0) 공통 파라미터 & 유틸
# ===============================
wavelength = 1550e-9                  # [m]
k0 = 2*np.pi/wavelength

# --- 감마 / 스케일 옵션 ---
gamma = 0.6            # 감마 보정 (작을수록 어두운 영역이 더 밝아짐)
shared_scale = False    # 두 imshow가 같은 컬러바 스케일을 쓰도록 할지 여부

# --- 공간 그리드 (개구 기반) ---
Nx = 2048
aperture_radius = 1.0e-3              # [m]
x = np.linspace(-aperture_radius, aperture_radius, Nx)
dx = x[1]-x[0]
R2 = x**2

# --- 렌즈 위상 (하이퍼볼릭: 구면파 정확식) ---
f_lens = 10e-3                        # [m] 유효 초점거리
# thin lens (parabolic) 대신 아래 하이퍼볼릭 위상 사용
# phi(x) = -k0*(sqrt(f^2 + x^2) - f)
phi = -k0 * (np.sqrt(f_lens**2 + R2) - f_lens)

# --- 입사각 설정 (필요 시 틸트 입사) ---
theta_deg = 0.0
theta_rad = np.deg2rad(theta_deg)

# --- 초기 필드 (원형 개구 + 틸트 위상) ---
aperture = (R2 <= aperture_radius**2)
incident_phase = np.exp(1j*k0*x*np.sin(theta_rad))
E0 = aperture * incident_phase

# --- ASM 전파 함수 (1D) ---
def angular_spectrum_1d(E_in, wavelength, dz, dx, n_medium=1.0):
    k0 = 2*np.pi/wavelength
    fx = np.fft.fftfreq(E_in.size, d=dx)
    kx = 2*np.pi*fx
    kz = np.sqrt((n_medium*k0)**2 - kx**2 + 0j)     # Evanescent 포함 0j 안전
    H = np.exp(1j*kz*dz)
    return np.fft.ifft(np.fft.fft(E_in) * H)

# ===============================
# 1) 스페이스플레이트 정의
# ===============================

#후보1-1 (NA : 0.1/ R:5.43 / fit : -1.1225 / I : 0.5098 / MSE : 1.1225e-3 / After : 154 / time : 0:5)
#d_list = [inf,54.27359713064605, 214.5758851352962, 228.82186907553708, 505.61237899355837, 72.89937169209226, 145.6466873093514, 266.9522517831467, 768.6236377370074, 296.98955628643324, 560.3614563936794, 1101.7356361037596, 162.26722551353214, 26.461437340167613, 29.441050099527924, 474.70678449786885, 1138.6221831649937, 29.691711908547703, 208.65714708558147, 719.0998037554577, 169.88575080462894, 89.04041557036426, 651.6463436427456, 57.13929927065206, 769.0082758675162, 311.58990425601564, 139.9565084911539, 876.1811573786522, 608.311025279582, 54.27293027673704, 57.83205303708311, 249.7967013282436, inf]

#후보2-1 (NA : 0.1/ R:6.11 / fit : -0.385 / I : 0.5415 / mapMSE : 3.4186e-3/angleMSE : 9.6275e-2 / After : 124 / time : 0:27)
#d_list = [inf,175.594615590951, 70.48430621874931, 709.7796937855082, 318.3367997238998, 113.92260060370053, 1056.0495731370227, 214.6496923764969, 228.77352185452244, 435.7099582023823, 228.09521796600785, 119.47778783306528, 403.7569682732755, 289.8425622656924, 377.22326965717446, 1086.5452174481086, 344.0017245593467, 208.30993703948857, 328.19882415080684, 18.63191709561039, 190.7374246168165, 294.53411204722937, 574.1055636006704, 830.3150384761818, 714.1463509709293, 254.0532666524952,inf]
#후보2-1 (NA : 0.1/ R:5.68 / fit : -0.114268 / I : 0.5368 / mapMSE : 1.2186e-3/angleMSE : 2.84e-2 / After : 399 / time : 0:12)
d_list= [inf,564.0214096127811,
    103.09230196316653,
    404.85290890928314,
    303.83992213200077,
    274.79693621263493,
    420.6127547810567,
    353.8489101286829,
    974.1655886304272,
    122.71921893559234,
    295.5440566132326,
    536.7343660632674,
    131.82361467176378,
    643.8991330019795,
    1079.2764627777003,
    578.5015476979983,
    60.54986241816496,
    229.35350766195245,
    116.70208340823153,
    475.03679199435214,
    214.01253031621573,
    487.66038546764884,
    84.54801475055746,
    206.32952034201378,
    429.5772517504288,
    836.1274513945332,inf]



def alternating_n_list(N):
    n1, n2 = 3.6211 + 1.26e-7j, 1.4596 + 0.000011139j
    n_list = [1]
    for i in range(N):
        n_list.append(n1 if i % 2 == 0 else n2)
    n_list.append(1)
    return n_list

num_layers = len(d_list) - 2
n_list = alternating_n_list(num_layers)



# 구조체 실제 두께(m)
structure_thickness_m = sum(d*1e-9 for d in d_list if np.isfinite(d))

# ===============================
# 2) Z 스케줄: z=0(렌즈면) → 자유공간 → (SP or 없음) → 포커스+여유
# ===============================
dz = 1e-6                              # [m] 전파 스텝
z_pre = 1e-3                           # 렌즈 뒤 자유공간(메탈렌즈면~구조체 사이)
z_air = 12e-3                          # 구조체 뒤 포커스+여유
Nz_pre = int(np.round(z_pre/dz))
Nz_air = int(np.round(z_air/dz))
Nz_struct = 1                          # SP 내부 z-전파 생략(단일 통과 모델)
Nz_total = Nz_pre + Nz_struct + Nz_air

# 절대 z축 (렌즈면이 z=0)
z = np.linspace(0.0, z_pre + structure_thickness_m + z_air, Nz_total)

# 결과 저장 (Intensity)
E_map_free = np.zeros((Nz_total, Nx), dtype=float)  # 자유공간만
E_map_sp   = np.zeros((Nz_total, Nx), dtype=float)  # SP 포함 (p-pol 맵)

# ===============================
# 3) 공통: z=0에서 렌즈 위상 부여
# ===============================
E_init = E0 * np.exp(1j*phi)

# ===============================
# 4) Case A: 자유공간만 전파
# ===============================
E = E_init.copy()
for i in range(Nz_pre):
    E = angular_spectrum_1d(E, wavelength, dz, dx, n_medium=1.0)
    E_map_free[i, :] = np.abs(E)**2

# SP 두께와 같은 인덱스에 기록만(동일 길이 맞춤)
E_map_free[Nz_pre, :] = np.abs(E)**2

for i in range(Nz_pre+1, Nz_total):
    E = angular_spectrum_1d(E, wavelength, dz, dx, n_medium=1.0)
    E_map_free[i, :] = np.abs(E)**2

# --- 추가: 정규화 전에 free on-axis 백업 ---
center_idx = Nx//2
I_on_free_pre = E_map_free[:, center_idx].copy()

# 정규화
E_map_free /= E_map_free.max() + 1e-16

# ===============================
# 5) Case B: 스페이스플레이트 포함 전파
# ===============================
E = E_init.copy()
for i in range(Nz_pre):
    E = angular_spectrum_1d(E, wavelength, dz, dx, n_medium=1.0)
    E_map_sp[i, :] = np.abs(E)**2

# 스페이스플레이트 단일 통과 (각 kx별 TMM 전송계수; p-pol 예시)
fx = np.fft.fftfreq(Nx, d=dx)
kx = 2*np.pi*fx
theta_kx = np.arcsin(np.clip(kx/k0, -1, 1))
t_coef = np.zeros(Nx, dtype=complex)
for i_k, th in enumerate(theta_kx):
    if np.abs(kx[i_k]) > k0 or np.isnan(th):
        t_coef[i_k] = 0.0
    else:
        try:
            t_coef[i_k] = coh_tmm('p', n_list, d_list, th, 1550)['t'] # 's'로 바꿔도 됨
        except Exception:
            t_coef[i_k] = 0.0

# --- 추가: s-pol 계산을 위해 Plate 입구면 필드 백업 ---
E_before_SP = E.copy()

E_fft = np.fft.fft(E)
E_fft *= t_coef
E = np.fft.ifft(E_fft)

E_map_sp[Nz_pre, :] = np.abs(E)**2

for i in range(Nz_pre+1, Nz_total):
    E = angular_spectrum_1d(E, wavelength, dz, dx, n_medium=1.0)
    E_map_sp[i, :] = np.abs(E)**2

# --- 추가: 정규화 전에 p-pol on-axis 백업 ---
I_on_sp_p_pre = E_map_sp[:, center_idx].copy()

# --- 추가: s-pol on-axis 계산(맵 전체 생성 없이) ---
t_coef_s = np.zeros(Nx, dtype=complex)
for i_k, th in enumerate(theta_kx):
    if np.abs(kx[i_k]) > k0 or np.isnan(th):
        t_coef_s[i_k] = 0.0
    else:
        try:
            t_coef_s[i_k] = coh_tmm('s', n_list, d_list, th, 1550)['t']
        except Exception:
            t_coef_s[i_k] = 0.0

I_on_sp_s_pre = np.zeros_like(I_on_sp_p_pre)
# Plate 전 구간은 동일
I_on_sp_s_pre[:Nz_pre] = E_map_sp[:Nz_pre, center_idx]

# Plate 통과 (s-pol 적용)
E_s = E_before_SP.copy()
E_fft_s = np.fft.fft(E_s)
E_fft_s *= t_coef_s
E_s = np.fft.ifft(E_fft_s)
I_on_sp_s_pre[Nz_pre] = np.abs(E_s[center_idx])**2

# Plate 이후 전파 (s-pol on-axis만)
for i in range(Nz_pre+1, Nz_total):
    E_s = angular_spectrum_1d(E_s, wavelength, dz, dx, n_medium=1.0)
    I_on_sp_s_pre[i] = np.abs(E_s[center_idx])**2

# 정규화
E_map_sp /= E_map_sp.max() + 1e-16

# ===============================
# 6) 포커스 위치 (절대 z좌표) - 각 케이스
# ===============================
z_peak_idx_free = np.argmax(E_map_free[:, center_idx])
z_peak_idx_sp   = np.argmax(E_map_sp[:,   center_idx])
z_peak_free = z[z_peak_idx_free]
z_peak_sp   = z[z_peak_idx_sp]

# --- 추가: s/p on-axis(정규화 전)로 각 포커스 위치도 계산 ---
z_peak_idx_sp_p = int(np.argmax(I_on_sp_p_pre))
z_peak_idx_sp_s = int(np.argmax(I_on_sp_s_pre))
z_peak_sp_p_mm  = z[z_peak_idx_sp_p]*1e3
z_peak_sp_s_mm  = z[z_peak_idx_sp_s]*1e3

# 주요 위치(mm)
z_lens_mm      = 0.0
z_sp_start_mm  = z_pre*1e3
z_sp_end_mm    = (z_pre + structure_thickness_m)*1e3
z_peak_free_mm = z_peak_free*1e3
z_peak_sp_mm   = z_peak_sp*1e3

# ===============================
# 7) 시각화: 2x2 (감마 보정 포함)
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 컬러 스케일 결정
if shared_scale:
    vmax_shared = np.percentile(np.r_[E_map_free.flatten(), E_map_sp.flatten()], 100)
    vmin_shared = 0.0
    norm_shared = PowerNorm(gamma=gamma, vmin=vmin_shared, vmax=vmax_shared)
    norm_free = norm_sp = norm_shared
else:
    norm_free = PowerNorm(gamma=gamma, vmin=0.0, vmax=np.percentile(E_map_free, 100))
    norm_sp   = PowerNorm(gamma=gamma, vmin=0.0, vmax=np.percentile(E_map_sp,   100))

# --- (좌상) 자유공간 imshow ---
im0 = axes[0,0].imshow(
    E_map_free.T,
    extent=[z[0]*1e3, z[-1]*1e3, x[0]*1e3, x[-1]*1e3],
    aspect='auto', cmap='inferno', origin='lower',
    norm=norm_free
)

axes[0,0].axvline(z_peak_free_mm, color='gray', lw=1.0, ls='--', label='Focus(Freespace)')
axes[0,0].set_title(f'Propagation w/o spaceplate (hyperbolic lens)',fontsize=15)
axes[0,0].set_xlabel('z (mm)',fontsize=13)
axes[0,0].set_ylabel('x (mm)',fontsize=13)
axes[0,0].set_xlim(9.5,10.5)
axes[0,0].set_ylim(-0.5,0.5)
axes[0,0].tick_params(axis='both',labelsize=13)
axes[0,0].legend(fontsize=9, loc='upper right')
cbar0 = fig.colorbar(im0, ax=axes[0,0])
cbar0.set_label('|E|^2 (normalized)')

# --- (우상) 스페이스플레이트 imshow ---
im1 = axes[0,1].imshow(
    E_map_sp.T,
    extent=[z[0]*1e3, z[-1]*1e3, x[0]*1e3, x[-1]*1e3],
    aspect='auto', cmap='inferno', origin='lower',
    norm=norm_sp
)
axes[0,1].axvline(z_peak_sp_mm,  color='yellow', lw=1.0, ls='--', label='Focus (Spaceplate)')
axes[0,1].axvline(z_peak_free_mm, color='gray', lw=1.0, ls='--', label='Focus(Freespace)')
axes[0,1].set_title(f'Propagation w/ spaceplate (hyperbolic lens)',fontsize=15)
axes[0,1].set_xlabel('z (mm)',fontsize=13)
axes[0,1].set_ylabel('x (mm)',fontsize=13)
axes[0,1].set_xlim(9.5,10.5)
axes[0,1].set_ylim(-0.5,0.5)
axes[0,1].tick_params(axis='both',labelsize=13)
axes[0,1].legend(fontsize=9, loc='upper right')
cbar1 = fig.colorbar(im1, ax=axes[0,1])
cbar1.set_label('|E|^2 (normalized)')

# --- (좌하) 자유공간 1D on-axis ---
axes[1,0].plot(z*1e3, E_map_free[:, center_idx], lw=1.2)
axes[1,0].axvline(z_peak_free_mm, color='gray',   lw=1.0, ls='--', label='Focus (Freespace)')
axes[1,0].set_xlabel('z (mm)',fontsize=13)
axes[1,0].set_ylabel('Normalized Intensity',fontsize=13)
axes[1,0].set_xlim(9.5,10.5)
axes[1,0].set_title('Intensity w/o Spaceplate',fontsize=15)
axes[1,0].tick_params(axis='both',labelsize=13)
axes[1,0].legend(fontsize=9)

# --- (우하) 스페이스플레이트 1D on-axis: free vs s vs p (각자 정규화) ---
eps = 1e-16
free_norm = I_on_free_pre.max()   + eps
s_norm    = I_on_sp_s_pre.max()   + eps
p_norm    = I_on_sp_p_pre.max()   + eps

#axes[1,1].plot(z*1e3, I_on_free_pre/free_norm, lw=1.2, label='Intensity (Freespace)')
axes[1,1].plot(z*1e3, I_on_sp_s_pre/s_norm,ls='-' , color='blue', lw=1.2, label='SP, s-pol')
axes[1,1].plot(z*1e3, I_on_sp_p_pre/p_norm,ls='--' , color='red', lw=1.2, label='SP, p-pol')

axes[1,1].axvline(z_peak_free_mm, color='gray', lw=1.0, ls='--', label='Focus(Freespace)')
axes[1,1].axvline(z_peak_sp_s_mm, lw=1.0, ls='-', color='blue')
axes[1,1].axvline(z_peak_sp_p_mm, lw=1.0, ls='--', color='red')

axes[1,1].set_xlabel('z (mm)',fontsize=13)
axes[1,1].set_ylabel('Intensity',fontsize=13)
axes[1,1].set_xlim(9.5,10.5)
axes[1,1].set_title('Intensity w/ Spaceplate',fontsize=15)
axes[1,1].tick_params(axis='both',labelsize=13)
axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.show()


