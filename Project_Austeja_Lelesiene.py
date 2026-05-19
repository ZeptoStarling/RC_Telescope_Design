import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import j1

np.random.seed(20)

class Ray:
    def __init__(self, origin, direction, path_length=0.0):
        self.origin = np.array(origin)
        self.direction = direction / np.linalg.norm(direction)
        self.path_length = path_length
    
def GenerateRays(n, z, D, step_size, offset, f_sys):
    """
    Generates n parallel rays distributed within a circular aperture of diameter D at plane z. 

    The propagation direction is tilted in the Y-Z plane based on the arctan of the linear offset (the distance of a point on the focal plane, measured away from the optical axis) 
    and system focal length f_sys. The coordinates for the rays are chosen using random sampling from a uniform grid.
    If the sampling density (step_size) is too low to generate n points that fit in the aperture, the function recursively increases density until n
    rays that satisfy the conditions are generated. Generates rays that travel in negative z direction.
    
    Returns a list of n Ray objects.
    """

    x = np.arange(-D/2, D/2, step_size)
    y = np.arange(-D/2, D/2, step_size)

    xx, yy = np.meshgrid(x, y)
    
    mask = (xx*xx + yy*yy) < (D/2)*(D/2)
    filtered_x = xx[mask]
    filtered_y = yy[mask]
    if len(filtered_x) < n:
        return GenerateRays(n, z, D, step_size * 0.9, offset, f_sys)
    
    angle = math.atan2(offset, f_sys)
    indices = np.random.choice(len(filtered_x), size=n, replace=False)
    rays = []
    for i in indices:
        ray = Ray(np.array([filtered_x[i], filtered_y[i], z]), np.array([0, math.sin(angle), -math.cos(angle)]))
        rays.append(ray)
    return rays

def Reflection(direction, N):
    return direction - 2 * np.dot(N, direction) * N

class Mirror():
    def __init__(self, R, K, z):
        self.R = R
        self.K = K
        self.z = z

    def GetZ(self, x, y):
        r = math.sqrt(x*x + y*y)
        val = 1 - (1 + self.K) * (r*r) / (self.R*self.R)
        if val < 0: return 0.0
        return (r*r) / (self.R * (1 + math.sqrt(val)))
    
    def GetNormal(self, x, y):
        r = math.sqrt(x*x + y*y)
        val = 1 - (1 + self.K) * (r*r) / (self.R*self.R)
        if val <= 0: val = 1e-15
        slope = r / (self.R * math.sqrt(val))
        if r < 1e-15: return np.array([0, 0, np.sign(self.R) * -1.0])
        nx = -slope * x / r
        ny = -slope * y / r
        nz = 1.0
        N = np.array([nx, ny, nz]) * np.sign(self.R) * -1.0
        return N / np.linalg.norm(N)

def ReflectRay(ray, mirror):
    t = (mirror.z - ray.origin[2]) / ray.direction[2]
    for _ in range(5):
        pos = ray.origin + t * ray.direction
        new_z = mirror.z + mirror.GetZ(pos[0], pos[1]) 
        z_error = pos[2] - new_z
        t_adjustment = z_error / (-ray.direction[2])
        t += t_adjustment
    final_pos = ray.origin + t * ray.direction

    segment_length = np.linalg.norm(final_pos - ray.origin)
    new_path_length = ray.path_length + segment_length
    
    N = mirror.GetNormal(final_pos[0], final_pos[1])
    reflected_dir = Reflection(ray.direction, N)

    return Ray(final_pos, reflected_dir, path_length=new_path_length)

class Telescope():
    """ 
    Models an aplanatic two-mirror telescope. Supports both Ritchey-Chrétien Cassegrain and aplanatic gregorian
    configurations.

    Args:
        name (str): Name for the specific configuration (e.g., "Hubble").
        D (float): Aperture diameter (in mm).
        f_primary (float): Focal length of the primary mirror (in mm).
        F_sys (float):  The effective system focal ratio (e.g., 12). 
                        IMPORTANT! Currently is also used as a toggle to determine 
                        telescope type (negative vals for Gregorian).
        s (float, optional): primary-secondary separation (mm)
        k (float, optional): relative minimum secondary size

    Note:
        Please provide at least either s or k.
        The primary mirror of the system is at z = 0.
    """
    def __init__(self, name, D, f_primary, F_sys, *, s=None, k=None):
        if not s and not k:
            raise ValueError("You must provide at least either 's' (primary-secondary separation) or 'k' (relative minimum secondary size).")
        
        self.name = name
        self.f_primary = f_primary
        self.D = D
        self.F_sys = F_sys
        
        if not k:
            self.k = (f_primary - s) / f_primary
        else:
            self.k = k
        if not s:
            self.s = f_primary - (self.k * f_primary)
        else:
            self.s = s
        
        self.f_sys = self.F_sys * self.D
        self.m = self.f_sys / self.f_primary
        self.f_sensor_z = self.s - (self.m * (self.f_primary - self.s))
        self.R_primary = self.f_primary * 2
        self.R_secondary = (self.m * self.k * self.R_primary) / (self.m - 1)
        self.z_primary = 0
        self.z_secondary = self.s
        self.eta = (self.m + 1) * self.k - 1
        self.Rm = ((1 + self.eta) * self.R_primary * self.m**2) / (2 * (self.m + 1) * (self.m**2 - (self.m - 1) * self.eta))
        self.K_primary = -1.0 - (2.0 * self.k) / ((1.0 - self.k) * self.m**2)
        self.K_secondary = -((self.m + 1.0) / (self.m - 1.0))**2 - (2.0 * self.m) / ((1.0 - self.k) * (self.m - 1.0)**3)

        self.primary = Mirror(self.R_primary, self.K_primary, self.z_primary)
        self.secondary = Mirror(self.R_secondary, self.K_secondary, self.z_secondary)

        print(f"Telescope name: {self.name}")
        print(f"The conics (K1, K2): {self.K_primary:.6f}, {self.K_secondary:.4f}")
        print(f"secondary magnification (m): {self.m:.3f}")
        print(f"relative minimum secondary size (k): {self.k:.3f}")
        print(f"back focal distance (eta): {self.eta:.4f}")
        print(f"best image surface curvature radius (Rm): {self.Rm:.2f}")
        
    def GetPointAtFocalPlane(self, ray, sensor_z):
        ray_to_sec = ReflectRay(ray, self.primary)
        ray_to_focus = ReflectRay(ray_to_sec, self.secondary)
        t = (sensor_z - ray_to_focus.origin[2]) / ray_to_focus.direction[2]
        x = ray_to_focus.origin[0] + ray_to_focus.direction[0] * t
        y = ray_to_focus.origin[1] + ray_to_focus.direction[1] * t
        return x, y

def DampEdge(arr, frac=0.15):
    """
    Smoothly tapers the outer 15% edges of a real-space grid toward zero 
    using a raised-cosine window, used in jinc-FT trick.
    """
    ny, nx = arr.shape

    x = np.fft.fftshift(np.fft.fftfreq(nx))
    y = np.fft.fftshift(np.fft.fftfreq(ny))
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)


    rr_norm = rr / 0.5

    taper = np.ones_like(rr_norm)
    start = 1.0 - frac
    mask = rr_norm > start

    taper[mask] = 0.5 * (
        1 + np.cos(np.pi * (rr_norm[mask] - start) / frac)
    )

    taper[rr_norm >= 1.0] = 0

    return arr * taper
def BuildSmoothedPupil(xx, yy, R_outer, R_inner=0):
    """
    Generates a smoothed pupil amplitude via the jinc-FT trick.
    Evaluates the analytical Fourier transform (Jinc) of a disk in real space,
    damps the outer 15% region, and returns to the discrete pupil plane.
    """
    N = xx.shape[0]
    dx = xx[0, 1] - xx[0, 0]
    
    freq_x = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fxx, fyy = np.meshgrid(freq_x, freq_x)
    fr = np.sqrt(fxx**2 + fyy**2)
    
    def analytical_jinc_circle(R, radius_grid):
        arg = 2 * np.pi * R * radius_grid
        val = np.zeros_like(radius_grid)
        mask = arg > 1e-12
        val[mask] = 2 * j1(arg[mask]) / arg[mask]
        val[~mask] = 1.0
        return val * (np.pi * R**2)

    real_space_field = analytical_jinc_circle(R_outer, fr)
    if R_inner > 0:
        real_space_field -= analytical_jinc_circle(R_inner, fr)
        
    smoothed_real_space = DampEdge(real_space_field, frac=0.15)
    
    recovered_pupil = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(smoothed_real_space)))
    pupil_amp = np.abs(recovered_pupil)
    
    return pupil_amp / np.max(pupil_amp)

def CalculateSinglePixelPSF(telescope, offset, curved_sensor=False, n_grid=128, wavelength=0.00055):
    z_curve_shift = (offset**2) / (2 * telescope.Rm) if curved_sensor else 0
    pixel_z = telescope.f_sensor_z + z_curve_shift
    pixel_pos = np.array([0, offset, pixel_z])
    reference_z = telescope.s + 1000 
    
    D_primary = telescope.D
    D_secondary = telescope.D * abs(telescope.k)
    
    x = (np.arange(n_grid) - n_grid // 2) * (D_primary / n_grid)
    y = (np.arange(n_grid) - n_grid // 2) * (D_primary / n_grid)
    xx, yy = np.meshgrid(x, y)

    R_outer = D_primary / 2
    R_inner = D_secondary / 2

    pupil_amp = BuildSmoothedPupil(xx, yy, R_outer=R_outer, R_inner=R_inner)

    path_lengths = np.zeros((n_grid, n_grid))
    mag_scale = D_secondary / D_primary

    for i in range(n_grid):
        for j in range(n_grid):
            r2 = xx[i,j]**2 + yy[i,j]**2
            if r2 > (D_primary/2)**2 or r2 < (D_secondary/2)**2:
                continue

            target = np.array([xx[i,j] * mag_scale, yy[i,j] * mag_scale, telescope.s])
            direction = target - pixel_pos
            ray = Ray(pixel_pos, direction)
            
            try:
                ray_sec = ReflectRay(ray, telescope.secondary)
                sec_r2 = ray_sec.origin[0]**2 + ray_sec.origin[1]**2
                if sec_r2 > (D_secondary/2)**2:
                    continue
                    
                ray_pri = ReflectRay(ray_sec, telescope.primary)
                pri_r2 = ray_pri.origin[0]**2 + ray_pri.origin[1]**2
                if pri_r2 > (D_primary/2)**2:
                    continue 
                    
                t_ref = (reference_z - ray_pri.origin[2]) / ray_pri.direction[2]
                final_pos = ray_pri.origin + t_ref * ray_pri.direction
                total_opl = ray_pri.path_length + np.linalg.norm(final_pos - ray_pri.origin)
                path_lengths[i, j] = total_opl
            except Exception:
                pass

    pupil_complex = np.zeros((n_grid, n_grid), dtype=complex)
    wavefront_waves = np.full((n_grid, n_grid), np.nan)
    valid_mask = path_lengths > 0
    
    if np.any(valid_mask):
        Z_data = path_lengths[valid_mask]
        X_coords = xx[valid_mask]
        Y_coords = yy[valid_mask]

        reference_opl = np.min(Z_data)
        opd_raw = path_lengths - reference_opl

        A_matrix = np.c_[X_coords, Y_coords, np.ones_like(X_coords)]
        C, _, _, _ = np.linalg.lstsq(A_matrix, opd_raw[valid_mask], rcond=None)

        tilt_plane = C[0] * xx + C[1] * yy + C[2]
        opd_full = opd_raw - tilt_plane
        reference_opd = np.mean(opd_full[valid_mask])
        
        for i in range(n_grid):
            for j in range(n_grid):
                if path_lengths[i, j] > 0:
                    opd = opd_full[i, j] - reference_opd
                    wavefront_waves[i, j] = opd / wavelength
                    phase = (2 * np.pi * opd) / wavelength
                    pupil_complex[i, j] = (pupil_amp[i, j] * np.exp(1j * phase))
                
    pad_size = 4 * n_grid
    padded_pupil = np.zeros((pad_size, pad_size), dtype=complex)
    start = (pad_size - n_grid) // 2
    padded_pupil[start:start+n_grid, start:start+n_grid] = pupil_complex
    
    
    psf_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_pupil)))
    psf_intensity = np.abs(psf_field)**2
    
    if np.max(psf_intensity) > 0:
        psf_intensity /= np.sum(psf_intensity)
        
    return pupil_complex, psf_intensity, wavefront_waves

test_telescopes = [
    {"name": "Hubble",                      "D": 2400, "f_primary": 5520, "F_sys": 24, "s": 4906.5, "k": None},
    {"name": "f_3_12_aplanatic_Gregorian",  "D": 300,  "f_primary": 900,  "F_sys": -12, "s": None,   "k": -0.417},
]

curved_sensor_options = [True, False]
wavelengths_to_test = [0.00065, 0.00045]  # 650 nm (red) and 450 nm (blue)

for surface_type in curved_sensor_options:
    for specs in test_telescopes:
        offsets = [0, 5, 10, 15]
        suffix = "curved" if surface_type else "flat"
        telescope = Telescope(**specs)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes = axes.flatten()
        ax_limit = 0.01
        
        for ax_idx, offset in enumerate(offsets):
            rays = GenerateRays(100, telescope.s + 1000, telescope.D, 1, offset, telescope.f_sys)
            z_curve_shift = (offset**2) / (2 * telescope.Rm) if surface_type else 0
            current_f_sensor_z = telescope.f_sensor_z + z_curve_shift
            
            x_pts, y_pts = [], []
            for ray in rays:
                _x, _y = telescope.GetPointAtFocalPlane(ray, current_f_sensor_z)
                x_pts.append(_x)
                y_pts.append(_y)
            axes[ax_idx].scatter(x_pts, y_pts, s=5, c='red')
            axes[ax_idx].set_title(f"Offset={offset}mm")
            axes[ax_idx].set_xlabel("x (mm)")
            axes[ax_idx].set_ylabel("y (mm)")
            axes[ax_idx].grid(True)
            axes[ax_idx].set_xlim(-ax_limit, ax_limit)
            axes[ax_idx].set_ylim(offset - ax_limit, offset + ax_limit)
            axes[ax_idx].set_aspect('equal')
            axes[ax_idx].xaxis.set_major_locator(ticker.MultipleLocator(0.005))
            axes[ax_idx].yaxis.set_major_locator(ticker.MultipleLocator(0.005))
            axes[ax_idx].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            axes[ax_idx].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            
        plt.tight_layout()
        plt.savefig(f"{telescope.name}_spot_diagram_{suffix}.png")
        plt.close()

        for wavelength in wavelengths_to_test:
            fig, axes = plt.subplots(2, len(offsets), figsize=(16, 8))
            fig.suptitle(f"Full Field PSF ({wavelength*1e6:.0f} nm): {telescope.name} ({suffix.capitalize()})", fontsize=16)

            n_grid = 256
            pad_size = 4 * n_grid
            center = pad_size // 2
            
            dx_pupil = telescope.D / n_grid 
            dx_psf = (wavelength * abs(telescope.f_sys)) / (pad_size * dx_pupil)
            
            fixed_half_width_mm = 0.075 
            zoom = int(fixed_half_width_mm / dx_psf)
            
            actual_half_width_mm = zoom * dx_psf
            extent_bounds = [-actual_half_width_mm, actual_half_width_mm, -actual_half_width_mm, actual_half_width_mm]

            for idx, offset in enumerate(offsets):
                pupil, psf, wavefront_waves = CalculateSinglePixelPSF(
                    telescope, offset=offset, curved_sensor=surface_type, n_grid=n_grid, wavelength=wavelength
                )
        
                ax_wave = axes[0, idx]
                im_w = ax_wave.imshow(wavefront_waves, cmap='coolwarm', vmin=-0.25, vmax=0.25)
                ax_wave.set_title(f"Wavefront (Offset: {offset}mm)")
                ax_wave.axis('off')
                
                ax_psf = axes[1, idx]
                ax_psf.imshow(
                    psf[center-zoom:center+zoom, center-zoom:center+zoom], 
                    cmap='inferno',
                    extent=extent_bounds
                )
                ax_psf.set_title(f"PSF Profile (mm)")
                ax_psf.axis('on')
                ax_psf.set_xlabel("x (mm)")
                ax_psf.set_ylabel("y (mm)")
                
            plt.tight_layout()
            plt.savefig(f"{telescope.name}_psf_{suffix}_{wavelength}.png")
            plt.close()