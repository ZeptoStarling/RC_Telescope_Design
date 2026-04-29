import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.random.seed(20)
#This uses repurposed code from previous homework 
class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = direction / np.linalg.norm(direction)
    
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
        return GenerateRays(n, z, D, step_size * 0.9,offset,f_sys)
    
    angle = math.atan2(offset,f_sys)
    indices = np.random.choice(len(filtered_x), size=n, replace=False)
    rays = []
    for i in indices:
        ray = Ray(np.array([filtered_x[i],filtered_y[i],z]), np.array([0, math.sin(angle), -math.cos(angle)]))
        rays.append(ray)
    return rays

def Reflection(direction,N):
    return direction - 2* np.dot(N,direction) * N

class Mirror():
    # Now I must set the z coordinate of the mirror, as there is two of them
    def __init__(self, R, K, z):
        self.R = R
        self.K = K
        self.z = z

    def GetZ(self, x, y):
        # self.z is the coordinate of the vertex of the mirror, while this calculates the z offset on the curved surface of the mirror at x, y
        r = math.sqrt(x*x + y*y)
        return (r*r)/(self.R*(1+math.sqrt(1-(1+self.K)*(r*r)/(self.R*self.R))))
    
    def GetNormal(self, x,y):
        r = math.sqrt(x*x + y*y)
        slope = r/(self.R*math.sqrt(1-(1+self.K)*(r*r)/(self.R*self.R)))
        N = [0,0,0]
        if r < 1e-15: return np.array([0,0,np.sign(self.R)*-1])
        nx = -slope*x/r
        ny = -slope*y/r
        nz = 1.0
        N = np.array([nx, ny, nz]) * np.sign(self.R)*-1
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
    N = mirror.GetNormal(final_pos[0],final_pos[1])
    reflected_dir = Reflection(ray.direction, N)
    return Ray(final_pos, reflected_dir)

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
            self.s = f_primary - (k*f_primary)
        else:
            self.s = s
        
        self.f_sys = self.F_sys * self.D
        self.m = self.f_sys / self.f_primary
        self.f_sensor_z = self.s - (self.m * (self.f_primary - self.s)) #"nominal" focal point
        self.R_primary = self.f_primary*2
        self.R_secondary = (self.m * self.k * self.R_primary) / (self.m - 1)
        self.z_primary = 0
        self.z_secondary = self.s
        self.eta = (self.m + 1) * self.k - 1
        self.Rm = ((1 + self.eta) * self.R_primary * self.m**2) / (2 * (self.m + 1) * (self.m**2 - (self.m-1)*self.eta))
        self.K_primary = -1.0 - (2.0 * self.k) / ((1.0 - self.k) * self.m**2)
        self.K_secondary = -((self.m + 1.0) / (self.m - 1.0))**2 - (2.0 * self.m) / ((1.0 - self.k) * (self.m - 1.0)**3)

        print(f"Telescope name: {self.name}")
        print(f"The conics (K1, K2): {self.K_primary:.6f}, {self.K_secondary:.4f}")
        print(f"secondary magnification (m): {self.m:.3f}")
        print(f"relative minimum secondary size (k): {self.k:.3f}")
        print(f"back focal distance (eta): {self.eta:.4f}")
        print(f"best image surface curvature radius (Rm): {self.Rm:.2f}")

        self.primary = Mirror(self.R_primary, self.K_primary, self.z_primary)
        self.secondary = Mirror(self.R_secondary, self.K_secondary, self.z_secondary)

    def GetPointAtFocalPlane(self, ray, sensor_z):
        """
        Gets the coordinates where a ray intersects the sensor plane located on the z axis at sensor_z.
        However, sensor_z gets varied for different offsets, to simulate the curved surface of best focus.
        """
        ray_to_sec = ReflectRay(ray, self.primary)
        ray_to_focus = ReflectRay(ray_to_sec, self.secondary)
        t = (sensor_z-ray_to_focus.origin[2])/ray_to_focus.direction[2]
        x = ray_to_focus.origin[0] + ray_to_focus.direction[0]*t
        y = ray_to_focus.origin[1] + ray_to_focus.direction[1]*t

        return x, y

test_telescopes = [
    {"name": "Hubble",                      "D": 2400, "f_primary": 5520, "F_sys": 24, "s": 4906.5, "k": None},
    {"name": "f_3_12_aplanatic_Gregorian",  "D": 300,  "f_primary": 900,  "F_sys": -12, "s": None,   "k": -0.417},
]

for specs in test_telescopes:
    n_rays = 100
    offsets = [0, 5, 10, 15]
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    axes = axes.flatten()
    ax = 0
    ax_limit = 0.01
    
    telescope = Telescope(**specs)
    #Produce spot diagrams
    for offset in offsets:
        rays = GenerateRays(100, telescope.s + 1000, telescope.D, 1, offset, telescope.f_sys)

        z_curve_shift = (offset**2) / (2 * telescope.Rm) #to compensate for field curvature
        current_f_sensor_z = telescope.f_sensor_z + z_curve_shift

        x = []
        y = []
        for ray in rays:
            _x,_y = telescope.GetPointAtFocalPlane(ray,current_f_sensor_z)
            x.append(_x)
            y.append(_y)
        axes[ax].scatter(x, y, s=5, c='red')
        axes[ax].set_title(f"Offset={offset}mm")
        axes[ax].set_xlabel("x (mm)")
        axes[ax].set_ylabel("y (mm)")
        axes[ax].grid(True)
        axes[ax].set_xlim(-ax_limit, ax_limit)
        axes[ax].set_ylim(offset - ax_limit, offset + ax_limit)
        axes[ax].set_aspect('equal')
        ax += 1
    plt.tight_layout(pad=4)
    plt.savefig(f"{telescope.name}_spot_diagram.png")
    plt.show()