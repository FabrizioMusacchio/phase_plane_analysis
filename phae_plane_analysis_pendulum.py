# %% IMPORT
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% EXAMPLE: PENDULUM
g = 9.81  # acceleration due to gravity in m/s^2
l = 1.0   # length of the pendulum rope in meters

# define the function for the phase portrait:
def pendulum_phase_portrait(t, z):
    theta, v = z  # theta is the angle, v is the angular velocity
    dtheta_dt = v
    dv_dt = -(g / l) * np.sin(theta)
    return [dtheta_dt, dv_dt]

# angles and speeds for phase portrait:
theta_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
v_vals = np.linspace(-10, 10, 100)
Theta, V = np.meshgrid(theta_vals, v_vals)
U = Theta, V = np.meshgrid(theta_vals, v_vals)
dTheta_dt, dV_dt = pendulum_phase_portrait(None, [Theta, V])

# calculate the trajectory for the pendulum:
Z0 = [1.0, 0] # initial conditions
t_span = [0, 10] # time span
t_eval = np.linspace(*t_span, 1000) # time points for the solution
sol = solve_ivp(pendulum_phase_portrait, t_span, Z0, t_eval=t_eval) # solve the ODE

# potential energy:
theta_energy = np.linspace(-2*np.pi, 2*np.pi, 400)
U_energy = l * (1 - np.cos(theta_energy))

# plot potential energy:
fig = plt.figure(1, figsize=(6, 4))
plt.plot(theta_energy, U_energy, 'r-')
plt.title('Potential energy of the pendulum')
plt.xlabel('Theta (rad)')
plt.ylabel('Potential energy (joules)')
plt.tight_layout()
plt.savefig('figures/pendulum_phase_energy.png', dpi=120)
plt.show()

# plot phase portrait:
fig = plt.figure(1, figsize=(6, 4))
speed = np.sqrt(dTheta_dt**2 + dV_dt**2)
#ax[1].streamplot(Theta, V, dTheta_dt, dV_dt, density=1.5)
plt.streamplot(Theta, V, dTheta_dt, dV_dt, color=speed, cmap='cool', density=1.5)
# plot the trajectory:
plt.plot(sol.y[0], sol.y[1], '-', label=f'Trajectory for $(\Theta_0,v_0)$=({Z0[0]},{Z0[1]})', c='r', lw=2)
# plot nullclines:
# the x-nullcline is just the y-axis since dt x=0 -> y=dt Theta=0:
plt.axhline(0, color='red', linestyle='--', label='x-nullcline ($\dot{x}=0$, $y=0$)')
# the y-nullcline is the set of points where dt y=0, i.e., x=n*pi for n in Z:
for n in range(-2, 3):  # for n from -2 to 2 to cover the relevant points
    plt.axvline(n * np.pi, color='blue', linestyle='--', label='y-nullcline ($\dot{y}=0$, $x=n\pi$)' if n == -2 else "")

# calculate and plot the separatrix for points near the saddle points:
epsilon = 1e-8  # small push
for theta0 in [ -3.0*np.pi + epsilon, 3.0*np.pi - epsilon]:
    sol_separatrix = solve_ivp(pendulum_phase_portrait, [-10, 10], [theta0, 0], 
                               t_eval=np.linspace(-10, 10, 1000))
    plt.plot(sol_separatrix.y[0], sol_separatrix.y[1], 'k--', lw=1.5, 
                 label='Separatrix' if theta0 == -3.0*np.pi + epsilon else "")

plt.xlim(-2*np.pi, 2*np.pi)
plt.legend(frameon=True, loc='upper right', fontsize=10)
plt.title('Phase portrait of the pendulum')
plt.xlabel('Theta (rad)')
plt.ylabel('Angular velocity (rad/s)')
plt.tight_layout()
plt.savefig(f'figures/pendulum_phase_portrait_z{Z0[0]}_{Z0[1]}.png', dpi=120)
plt.show()
# %% END