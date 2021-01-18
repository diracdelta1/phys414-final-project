# Yusuf Sarıyar, 18.01.2021
# Phys414 Final Project, Newton Part
import numpy as np
import matplotlib.pyplot as plt
from pylab import polyfit

csv = np.genfromtxt('white_dwarf_data.csv', delimiter=",",skip_header=1)
log_g = csv[:,1] # stores log10(g) data
mass = csv[:,2] # stores mass data

g = np.power(10, log_g)/100 # g in mks units
gravity_const = 6.6743e-11 # gravity constant G in mks units
solar_mass = 1.989e30 # kg/solar mass (for unit transformation)
earth_radius = 6.371e6 # meters

radius = np.sqrt(gravity_const*mass*solar_mass/g)/earth_radius # stores r values

# Plotting
plt.plot(radius, mass, 'bs',markersize=1)
plt.title("M vs R Graph")
plt.xlabel("Radius (in average Earth radius)")
plt.ylabel("Mass (in Solar Mass)")
plt.grid()
plt.savefig('M_vs_R.png',dpi=300,transparent=True)

# Choosing small masses for fitting
fit_mass = []
fit_radius = []

for i in range(len(mass)):
    if (mass[i]<0.5):
        fit_mass = np.append(fit_mass,mass[i])
        fit_radius = np.append(fit_radius,radius[i])

# Fitting M=BR^gamma where gamma=(3-n)/(1-n)
# ln(M)=ln(B)+gamma*ln(R)
ln_radius = np.log(fit_radius)
ln_mass = np.log(fit_mass)

gamma,ln_B = polyfit(ln_radius, ln_mass, 1)
B = np.exp(ln_B)
n_star = (3-gamma)/(1-gamma)
n_star, B

# Lane-Emden Solver
def lane_emden_solver(n):

    dxi = 0.01 # step distances
    N = 100000

    xi = 0.0001 
    theta = 1.0
    f = 0.0
    theta_sol = []
    xi_sol = []

    for i in range(N):
        f += -xi**2*theta**n*dxi
        theta += f/xi**2*dxi
        xi += dxi
        theta_sol.append(theta)
        xi_sol.append(xi)
        if (theta_sol[i]*theta_sol[i-1]<0): # stopping condition where we found a solution
            break
            
    der_theta_xi = theta_sol[i]-theta_sol[i-1]/dxi # derivative of theta wrt xi
    
    return xi, der_theta_xi

xi_n, der_theta_xi_n = lane_emden_solver(1.5)

rho_c = -(fit_mass*xi_n)/(4*np.pi*fit_radius**3*der_theta_xi_n) # rho_c using the eq.10

# Plotting
plt.plot(fit_mass, rho_c, 'bs',markersize=1)
plt.title("rho_c vs M Graph for Small Masses")
plt.xlabel("Mass (in Solar Mass)")
plt.ylabel("rho_c")
plt.grid()
plt.savefig('rho_c_vs_M.png',dpi=300,transparent=True)
