# Yusuf Sarıyar, 18.01.2021
# Phys414 Final Project, Einstein Part
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def solve_tov(rho_c, var):
    
    def stop_condition(r, y): # where rho vanishes
        return y[2]
    stop_condition.terminal = True
    
    def find_rho(p, K = var): # finds rho for a given p
        rho = np.sqrt(p/K)
        return rho
 
    def tov(r, y):
        m = y[0] # mass
        v = y[1] # v
        p = y[2] # p
        baryonic_mass = y[3] # baryonic mass
        
        # der_x = derivative of x wrt r
        der_m = 4*np.pi*r**2*find_rho(p)
        der_v = (2*(m + 4*np.pi*r**3*p))/(r*(r-2*m))
        der_r = -0.5*(p+find_rho(p))*der_v
        der_baryonic_mass = 4*np.pi*(1-2*m/r)**(-1/2)*r**2*find_rho(p)
        
        return [der_m, der_v, der_r, der_baryonic_mass]
    
    p_c = var*rho_c**2
    y_initial = [0,0,p_c,0] 
    dt = 1e-10
    tspan = np.linspace(0+dt,100,1000000) # t=0 gives singularity hence excluded
    sol = solve_ivp(lambda r, y: tov(r, y),[tspan[0], tspan[-1]], y_initial,events=stop_condition) 
    
    return sol

sol = solve_tov(1,100)

# Plotting
plt.plot(sol.t, sol.y[0], 'bs',markersize=1)
plt.title("M vs R Graph")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.grid()
plt.savefig('M_vs_R_einstein.png',dpi=300,transparent=True)

# fractional binding energy 
delta = (sol.y[3]-sol.y[0])/sol.y[0]

# Plotting
plt.plot(np.delete(sol.t,0), np.delete(delta,0), 'bs',markersize=1)
plt.title("Fractional binding energy vs Radius Graph")
plt.xlabel("Radius")
plt.ylabel("Delta")
plt.grid()
plt.savefig('delta_vs_R.png',dpi=300,transparent=True)