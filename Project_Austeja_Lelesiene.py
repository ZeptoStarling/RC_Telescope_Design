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
    
def GenerateRays(n, z, D, step_size, offset):
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
        return GenerateRays(n, z, D, step_size * 0.9,offset)
    
    global f_sys
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
    
def CalculateK(f_sys, f_primary, k):
    """
    Calculates the conics for the primary and secondary mirrors.
    
    #m is secondary magnification - it describes how much the secondary mirror magnifies the image formed by the primary mirror.
    """
    m = f_sys / f_primary 

    K_primary = -1.0 - (2.0 * k) / ((1.0 - k) * m**2)
    K_secondary = -((m + 1.0) / (m - 1.0))**2 - (2.0 * m) / ((1.0 - k) * (m - 1.0)**3)


    return K_primary, K_secondary

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

def GetPointAtFocalPlane(ray, primary, secondary, sensor_z):
    """
    Gets the coordinates where a ray intersects the sensor plane located on the z axis at sensor_z.
    However, sensor_z gets varied for different offsets, to simulate the curved surface of best focus.
    """
    ray_to_sec = ReflectRay(ray, primary)
    ray_to_focus = ReflectRay(ray_to_sec, secondary)
    t = (sensor_z-ray_to_focus.origin[2])/ray_to_focus.direction[2]
    x = ray_to_focus.origin[0] + ray_to_focus.direction[0]*t
    y = ray_to_focus.origin[1] + ray_to_focus.direction[1]*t

    return x, y



#Hard-coded parameters for Hubble
# in mm
#--------------------
s = 4906.5 # Primary-Secondary separation
D = 2400 
f_primary = 5520 

F_sys = 24 # Effective system focal ratio
f_sys = F_sys * D
#--------------------
#Derived parameters
R_primary = f_primary*2
m = f_sys / f_primary
print(f"secondary magnification: {m:.3f} | Expected: 10.435")
f_sensor_z = s - (m * (f_primary - s)) #"nominal" focal point
k = (f_primary - s) / f_primary
print(f"relative minimum secondary size: {k:.3f} | Expected: 0.112")
R_secondary = (m * k * R_primary) / (m - 1)
z_primary = 0
z_secondary = s
eta = (m + 1) * k - 1
print(f"back focal distance (eta): {eta:.4f} | Expected: 0.2800")
Rm = ((1 + eta) * R_primary * m**2) / (2 * (m + 1) * (m**2 - (m-1)*eta))
print(f"best image surface curvature radius (Rm): {Rm:.2f} | Expected: 633.00")
K_primary, K_secondary = CalculateK(f_sys, f_primary, k)
print(f"The conics (K1, K2): {K_primary:.6f}, {K_secondary:.4f} | Expected: -1.002300, -1.4970")

#General setup
n_rays = 100
offsets = [0, 5, 10, 15]
fig, axes = plt.subplots(1, 4, figsize=(20, 10))
axes = axes.flatten()
ax = 0
ax_limit = 0.003

#Produce spot diagrams
for offset in offsets:
    rays = GenerateRays(100, s + 1000, D, 1, offset)
    primary = Mirror(R_primary, K_primary, z_primary)
    secondary = Mirror(R_secondary, K_secondary, z_secondary)
    z_curve_shift = (offset**2) / (2 * Rm) #to compensate for field curvature
    current_f_sensor_z = f_sensor_z + z_curve_shift

    x = []
    y = []
    for ray in rays:
        _x,_y = GetPointAtFocalPlane(ray,primary,secondary,current_f_sensor_z)
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
plt.savefig("Hubble_spot_diagram.png")
plt.show()