"""
Simple examples for understanding phase plane analysis, applied to the Van der Pol oscillator
using Liénard's transformation.

author: Fabrizio Musacchio
date: Feb 20, 2024

For reproducibility:

conda create -n phase_plane_analysis python=3.10
conda activate phase_plane_analysis
conda install mamba
mamba install numpy matplotlib scipy

"""
# %% IMPORT
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# set global font size for plots:
plt.rcParams.update({'font.size': 14})
# create a folder "figures" to save the plots (if it does not exist):
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% EXAMPLE: VAN DER POL OSCILLATOR
# define the Van der Pol oscillator model:
def van_der_pol(t, z, mu):
    x, y = z
    dxdt = mu * (x - (x**3)/3 - y) 
    dydt = (1/mu) * x
    return [dxdt, dydt]

# define the nullclines:
def x_nullcline(x, mu):
    return x-(1/3)*(x**3)
"""the y-nullcline is in this case just a vertical line at x=0 for all y;
we well plot it as a function of x for the sake of consistency with the other nullcline:
"""
def y_nullcline(x):
    return 0*x


# set time span:
eval_time = 100
t_iteration = 1000
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

# set initial conditions:
z0 = [0.5, 0.5] # [2, 0]

# set Van der Pol oscillator parameter:
mu = 2.0 # stable: >0, unstable: <0

# calculate the vector field:
mgrid_size = 8
x, y = np.meshgrid(np.linspace(-mgrid_size, mgrid_size, 15), 
                   np.linspace(-mgrid_size, mgrid_size, 15))
u = mu * (x - (x**3)/3 - y) 
v = (1/mu) * x

# calculating the trajectory for the Van der Pol oscillator:
sol_stable = solve_ivp(van_der_pol, t_span, z0, args=(mu,), t_eval=t_eval)

# define the x-array for the nullclines:
x_null = np.arange(-mgrid_size,mgrid_size,0.001)

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
#plt.quiver(x, y, u, v, color='gray')  # vector field
#plt.streamplot(x, y, u, v, color='gray', density=2.5)
# plot the streamline plot colored by the speed of the flow:
speed = np.sqrt(u**2 + v**2)
plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, x_nullcline(x_null, mu)  , '.', c="darkturquoise", markersize=2)
plt.plot(y_nullcline(x_null), x_null  , '.', c="darkturquoise", markersize=2)
plt.plot(sol_stable.y[0], sol_stable.y[1], 'r-', lw=3,
         label=f'Trajectory for $\mu$={mu}\nand $z_0$={z0}')  # trajectory
# indicate start point:
plt.plot(sol_stable.y[0][0], sol_stable.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol_stable.y[0][-1], sol_stable.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
# indicate the direction of the trajectory's last point with an arrow:
""" plt.arrow(sol_stable.y[0][-2], sol_stable.y[1][-2], 
          sol_stable.y[0][-1] - sol_stable.y[0][-2], 
          sol_stable.y[1][-1] - sol_stable.y[1][-2], 
          head_width=0.5, head_length=0.5, fc='fuchsia', ec='fuchsia') """
plt.title('phase plane plot: Van der Pol oscillator')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right') #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
#plt.xlim(-mgrid_size, mgrid_size)
plt.ylim(-mgrid_size, mgrid_size)
plt.tight_layout()
plt.savefig(f'figures/van_der_pol_oscillator_lienard_z_{z0[0]}_{z0[1]}_mu_{mu}.png', dpi=120)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(sol_stable.t, sol_stable.y[0], 'b-', lw=2, label='$x(t)$')
plt.title(f'$x(t)$, $\mu$={mu}, $z_0$={z0}')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend(loc='best')
plt.grid(True)
# remove box on the right and top:
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(f'figures/van_der_pol_oscillator_lienard_x_t_{z0[0]}_{z0[1]}_mu_{mu}.png', dpi=120)
plt.show()
# %% END
