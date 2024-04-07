"""
Phase plane analysis of the FitzHugh-Nagumo model.

author: Fabrizio Musacchio
date: Mar 24, 2024

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
# %% FITZHUGH-NAGUMO MODEL
# define the FitzHugh-Nagumo model:
def fitzhugh_nagumo(t, z, mu, a, b, I_ext):
    v, w = z
    dvdt = mu * (v - (v**3) / 3 - w + I_ext)
    dwdt = (1 / mu) * (v + a - b * w)
    return [dvdt, dwdt]

# define the nullclines:
def v_nullcline(v, I_ext):
    return v - (v**3) / 3 + I_ext
def w_nullcline(v, a, b):
    return (1 / b) * (v + a)

# set time span:
eval_time   = 100
t_iteration = 1000 # set to 5000 for the detailed trajectory plot; otherwise 1000 is sufficient
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

# set initial conditions:
z0 = [-1, -0.8] # [-1, -0.8] or [-1.2017543859649122, -0.6271929824561404]
mu = 2.0
a = 0.7
b = 0.8
I_ext = 0.25 # for ~0.34 to ~1.4, the system shows spiking behavior; 

# find the intersection of the nullclines for the given I_ext:
v_range = np.linspace(-3.5, 3.5, 400)
v_nullcline_w = v_nullcline(v_range, I_ext)
w_nullcline_v = w_nullcline(v_range, a, b)
intersection = np.argmin(np.abs(v_nullcline_w - w_nullcline_v))
# reset z0 to the intersection point:
#z0 = [v_range[intersection], w_nullcline_v[intersection]]
print([v_range[intersection], w_nullcline_v[intersection]])

# calculate the vector field:
mgrid_size = 3
x, y = np.meshgrid(np.linspace(-mgrid_size, mgrid_size, 15), 
                   np.linspace(-mgrid_size, mgrid_size, 15))
u = mu * (x - (x**3)/3 - y + I_ext)
v = (1/mu) * (x + a - b * y)

# calculating the trajectory for the Van der Pol oscillator:
sol = solve_ivp(fitzhugh_nagumo, t_span, z0, args=(mu, a, b, I_ext), t_eval=t_eval)

# define the x-array for the nullclines:
x_null = np.arange(-mgrid_size,mgrid_size,0.001)

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
# plot the streamline plot colored by the speed of the flow:
speed = np.sqrt(u**2 + v**2)
plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, v_nullcline(x_null, I_ext), '.', c="darkturquoise", markersize=2)
plt.plot(x_null, w_nullcline(x_null, a, b), '.', c="darkturquoise", markersize=2)
plt.plot(sol.y[0], sol.y[1], 'r-', lw=3, label=f'Trajectory, $z_0$={np.round(z0,2)}')
""" # mark the sections of the trajectory with different colors:
# until argmax(sol.y[0]) in green:
plt.plot(sol.y[0][:np.argmax(sol.y[0])], 
         sol.y[1][:np.argmax(sol.y[0])], 'g-', lw=3, label='depolarization')
# after argmax(sol.y[0]), find sol.y[0] being the first time -1:
idx_repolarization = np.where(sol.y[0] < -1)[0][0]
plt.plot(sol.y[0][np.argmax(sol.y[0]):idx_repolarization], 
         sol.y[1][np.argmax(sol.y[0]):idx_repolarization],
         '-', c="orange", lw=3, label='repolarization')
# find the index after idx_repolarization, where sol.y[0] reaches global minimum:
idx_hyperpolarization_fast = np.argmin(sol.y[0])
plt.plot(sol.y[0][idx_repolarization:idx_hyperpolarization_fast],
            sol.y[1][idx_repolarization:idx_hyperpolarization_fast],
            '--', c="purple", lw=3, label='hyperpolarization (fast)')
# find the index after idx_hyperpolarization_fast, where sol.y[0] is the first time >-1:
idx_hyperpolarization_slow = np.where(sol.y[0][idx_hyperpolarization_fast:] > -1)[0][0] + idx_hyperpolarization_fast
plt.plot(sol.y[0][idx_hyperpolarization_fast:idx_hyperpolarization_slow], 
         sol.y[1][idx_hyperpolarization_fast:idx_hyperpolarization_slow],
            '-', c="purple", lw=3, label='hyperpolarization (slow)')
plt.plot(sol.y[0][idx_hyperpolarization_slow:], sol.y[1][idx_hyperpolarization_slow:],
            '-', c="tomato", lw=3, label='resting state') """
# indicate start point:
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
# indicate the direction of the trajectory's last point with an arrow:
plt.title(f'phase plane plot: FitzHugh-Nagumo model\na: {a}, b: {b}, $\mu$: {mu}, $I_{{ext}}$: {I_ext}')
plt.xlabel('v')
plt.ylabel('w')
#plt.legend(loc='lower right', fontsize=13) #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlim(-mgrid_size, mgrid_size)
plt.ylim(-mgrid_size, mgrid_size)
#plt.xlim(-1.215, -1.18)
#plt.ylim(-0.63, -0.62)
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_model_z_{z0[0]}_{z0[1]}_mu_{mu}_I_{I_ext}.png', dpi=120)
plt.show()

# plot v over time to visualize the voltage curve / spiking behavior
plt.figure(figsize=(8, 5))
plt.figure(figsize=(6, 6))
plt.plot(sol.t, sol.y[0], 'b-', lw=2, label='Voltage $v(t)$')
""" # mark the sections of the trajectory with different colors as before:
plt.plot(sol.t[:np.argmax(sol.y[0])], 
         sol.y[0][:np.argmax(sol.y[0])], 'g-', lw=3, label='depolarization')
plt.plot(sol.t[np.argmax(sol.y[0]):idx_repolarization],
            sol.y[0][np.argmax(sol.y[0]):idx_repolarization],
            '-', c="orange", lw=3, label='repolarization')
# find the index after idx_repolarization, where sol.y[0] reaches global minimum:
idx_hyperpolarization_fast = np.argmin(sol.y[0])
plt.plot(sol.t[idx_repolarization:idx_hyperpolarization_fast],
            sol.y[0][idx_repolarization:idx_hyperpolarization_fast],
            '--', c="purple", lw=3, label='hyperpolarization (fast)')
# find the index after idx_hyperpolarization_fast, where sol.y[0] is the first time >-1:
idx_hyperpolarization_slow = np.where(sol.y[0][idx_hyperpolarization_fast:] > -1)[0][0] + idx_hyperpolarization_fast
plt.plot(sol.t[idx_hyperpolarization_fast:idx_hyperpolarization_slow], 
         sol.y[0][idx_hyperpolarization_fast:idx_hyperpolarization_slow],
            '-', c="purple", lw=3, label='hyperpolarization (slow)')
plt.plot(sol.t[idx_hyperpolarization_slow:], sol.y[0][idx_hyperpolarization_slow:],
            '-', c="tomato", lw=3, label='resting state') """
plt.xlim([0, 20])
plt.title(f'voltage curve: FitzHugh-Nagumo model\na: {a}, b: {b}, $\mu$: {mu}, $I_{{ext}}$: {I_ext}')
plt.xlabel('Time')
plt.ylabel('Voltage $v$')
plt.legend(loc='best', fontsize=13)
plt.grid(True)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_voltage_curve_{z0[0]}_{z0[1]}_mu_{mu}_I_{I_ext}.png', dpi=120)
plt.show()
# %% PLOT NULLCLINES FOR DIFFERENT I_EXT VALUES

# set time span:
eval_time   = 200
t_iteration = 2000
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

# set initial conditions:
z0 = [-1.2017543859649122, -0.6271929824561404]#[-1.2017543859649122, -0.6271929824561404]
mu = 2.0
a = 0.7
b = 0.8

# define the range of I_ext values for plotting different nullclines:
I_ext_values = [0, 0.30, 0.5, 0.6, 1.05, 1.4, 1.5, 1.9]

# define the v range for plotting:
v_range = np.linspace(-3.5, 3.5, 400)

# plot nullclines as a function of t, i.e., for selected varying I_ext values:
plt.figure(figsize=(7, 6))
# plot w-nullcline (only depends on v, a, and b, so it's constant)
w_nullcline_v = (1 / b) * (v_range + a)
plt.plot(v_range, w_nullcline_v, 'k--', label='w-nullcline')

# plot v-nullclines for different I_ext values:
for I_ext in I_ext_values:
    # calculate the v-nullcline for the current I_ext value:
    v_nullcline_w = v_range - (v_range**3) / 3 + I_ext
    # mark the intersection of the nullclines:
    intersection = np.argmin(np.abs(v_nullcline_w - w_nullcline_v))
    plt.plot(v_range[intersection], v_nullcline_w[intersection], 'ro', markersize=5,
             label="v-w intersection" if I_ext == 0 else None)
    # plot the v-nullcline:
    plt.plot(v_range, v_nullcline_w, label=f'v-nullcline for $I_{{ext}}={I_ext}$')
    
plt.title('FitzHugh-Nagumo nullclines for varying $I_{ext}$')
plt.xlabel('v')
plt.ylabel('w')
plt.legend(fontsize=12.5, loc='lower right')
plt.grid(True)
plt.xlim([-2.5, 2.5])
plt.ylim([-3, 3])
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_nullclines_varying_I_ext_{z0[0]}_{z0[1]}_mu_{mu}.png', dpi=120)
plt.show()
# %% TIME-DEPENDENT EXTERNAL CURRENT: CURRENT RAMPING
# define the FitzHugh-Nagumo model with time-dependent ramping I_ext:
def fitzhugh_nagumo_time_dependent_ramping(t, z, mu, a, b, I_ext_start, I_ext_end, eval_time):
    v, w = z
    # linearly ramp I_ext from I_ext_start to I_ext_end over the time span:
    I_ext = I_ext_start + (I_ext_end - I_ext_start) * (t / eval_time)
    dvdt = mu * (v - (v**3) / 3 - w + I_ext)
    dwdt = (1 / mu) * (v + a - b * w)
    return [dvdt, dwdt]

# set time span:
eval_time   = 200
t_iteration = 2000
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

# set parameters for I_ext ramping:
I_ext_start = 0.0  # Starting value of I_ext
I_ext_end = 1.9    # Ending value of I_ext

# set initial conditions:
#z0 = [1, 2]
z0 = [-1.2017543859649122, -0.6271929824561404]# same as before
mu = 2.0
a = 0.7
b = 0.8

# calculate the trajectory with time-dependent I_ext:
sol = solve_ivp(fitzhugh_nagumo_time_dependent_ramping, t_span, z0, 
                args=(mu, a, b, I_ext_start, I_ext_end, eval_time), t_eval=t_eval)

# calculate I_ext(t) for plotting:
I_ext_t = I_ext_start + (I_ext_end - I_ext_start) * (sol.t / eval_time)

# plot the voltage curve v(t):
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage $v(t)$', color=color)
ax1.plot(sol.t, sol.y[0], color=color, lw=2, label='Voltage $v(t)$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# instantiate a second y-axis for I_ext(t):
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('$I_{ext}(t)$', color=color)
ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$')
ax2.tick_params(axis='y', labelcolor=color)
# Remove box on the right and top for current axis
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(color)

lines,  labels  = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.title('Voltage curve for ramping external current')
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_voltage_I_ext_ramping_{z0[0]}_{z0[1]}_mu_{mu}.png', dpi=120)
plt.show()
# %% TIME-DEPENDENT EXTERNAL CURRENT: PULSED CURRENT
# define the FitzHugh-Nagumo model with time-dependent (pulsed) I_ext:
def fitzhugh_nagumo_time_dependent_pulse(t, z, mu, a, b, I_ext, pulse_time_ranges):
    v, w = z
    # set I_ext to zero except at pulse_time:
    I_ext_use = 0.0
    # add a pulse during pulse_time_range:
    for pulse_time_range in pulse_time_ranges:
        if (t >= pulse_time_range[0]) and (t <= pulse_time_range[1]):
            I_ext_use = I_ext
    dvdt = mu * (v - (v**3) / 3 - w + I_ext_use)
    dwdt = (1 / mu) * (v + a - b * w)
    return [dvdt, dwdt]

""" # re-define the v-nullclines:
def v_nullcline_time_dependent(t, v, I_ext, pulse_time_ranges):
    # set I_ext to zero except at pulse_time:
    I_ext_use = np.zeros_like(t)
    for time_i, time in enumerate(t):
        for pulse_time_range in pulse_time_ranges:
            if (time >= pulse_time_range[0]) and (time <= pulse_time_range[1]):
                I_ext_use[time_i] = I_ext
        #print(I_ext_use[time_i])
    result = v - (v**3) / 3 + I_ext_use 
    return result """
        
# set time span:
eval_time   = 50
t_iteration = 400
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

""" # define the x-array for the nullclines from -mgrid_size to mgrid_size with a spacing of t_iteration:
x_null = np.linspace(-mgrid_size, mgrid_size, t_iteration) """

# set parameters for I_ext pulsed:
I_ext = 1.0 # external current during pulse
#pulse_time_ranges = [[10,11], [9,10], [30,31], [37,38], [50,90]]
pulse_time_ranges = [[14,15]]
#pulse_time_ranges = [[10,11]] # [14,15] - [24,25]
#pulse_time_ranges = [[30,31], [35,36]] 
# calculate sol.t manually to get the correct time points for the delta pulse:
#sol_t = np.linspace(*t_span, t_iteration)
#pulse_time_ranges = [[sol_t[19],sol_t[31]]]
"""
Suppose we the time unit is in seconds. We set the time span to 50 seconds, and the time resolution
to 200 time steps. This means, that one second is resolved with 4 time steps. When we apply a current
pulse of 1 second by setting the delta_pulse_range, e.g., to [9,10], which corresponds to 1 second,
we actually apply a pulse lasting to time steps. By definition, this is not a delta pulse. 
Therefore, bear in mind, that a pulse of 1 second in the context of our simulation is not a delta pulse.
We can consider it as a short pulse instead.


Simulating a short pulsed peak (set pulse_time_range spanning only one second):

[10,11]: no ap
[9,10]: ap 

-> depending on the location on the trajectory in phase space, the system can either 
spike or not spike with a short pulse, i.e., a longer pulse is sometimes needed to 
trigger an action potential. This is another benefit of the model, as the simulated
sub-threshold oscillations that come along with the model due to a diminished limit cycle
behavior around the fixed point for low or zero I_ext values introduce another randomness
factor in the system, further mimicking the stochastic nature of real neurons.

Furthermore: The randomness in the system also affects the absolute amplitude of the
action potential, to some extent, i.e., a delta pulse at different times (e.g., [9,10], [30,31]) 
will lead to different action potential amplitudes, introducing another randomness factor in the 
simulation and thus a more realistic behavior of the model.

Prove, that during the absolute refractory period, the system cannot spike: 
Set initial pulse at [30,31] -> ap triggered. Another pulse at [33,34] -> no new ap triggered.
Same is tru for [34,35], [35,36], ... until [37,38] -> another ap is triggered. This is an 
important aspect of the model, as it mimics the refractory period of real neurons. It demonstrates
that it is important, *when* external stimuli reach the neuron, as the neuron cannot spike again

For constant step current, , e.g., [90,150], the system will spike continuously (for
a sufficiently high I_ext value), i.e., we get a train of action potentials. As soon as the external 
current is removed, the system returns to the resting state.
"""

# set initial conditions:
#z0 = [1, 2]
z0 = [-1.2017543859649122, -0.6271929824561404]# same as before
mu = 2.0
a = 0.7
b = 0.8

# calculate the trajectory with time-dependent I_ext:
sol = solve_ivp(fitzhugh_nagumo_time_dependent_pulse, t_span, z0, 
                args=(mu, a, b, I_ext, pulse_time_ranges), t_eval=t_eval)

# calculate I_ext(t) for plotting:
I_ext_t = np.zeros_like(sol.t)
for pulse_time_range in pulse_time_ranges:
    I_ext_t[(sol.t >= pulse_time_range[0]) & (sol.t <= pulse_time_range[1])] = I_ext

# print absolute minimum of the voltage curve:
print(f"Absolute minimum of the voltage curve: {np.min(sol.y[0])}")
print(f"Absolute maximum of the voltage curve: {np.max(sol.y[0])}")

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
# plot the streamline plot colored by the speed of the flow:
#speed = np.sqrt(u**2 + v**2)
#plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, v_nullcline(x_null, 0), '-', c="darkturquoise", lw=2, 
         label=f'v-nullcline for $I_{{ext}}=0$')
plt.plot(x_null, v_nullcline(x_null, I_ext), '--', c="darkturquoise", lw=2,
         label=f'v-nullcline for $I_{{ext}}={I_ext}$')
""" plt.plot(x_null, v_nullcline_time_dependent(sol.t, x_null, I_ext, pulse_time_ranges),
            '-', c="darkturquoise", lw=2) """
plt.plot(x_null, w_nullcline(x_null, a, b), '-', c="darkturquoise", lw=2)
plt.plot(sol.y[0], sol.y[1], 'r-', lw=3, label=f'Trajectory, $z_0$={np.round(z0,2)}')
# indicate start point:
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
# indicate the direction of the trajectory's last point with an arrow:
plt.title(f'phase plane plot: FitzHugh-Nagumo model\na: {a}, b: {b}, $\mu$: {mu},\n'+ \
          f"pulse intervals: {pulse_time_ranges}")
plt.xlabel('v')
plt.ylabel('w')
plt.legend(loc='lower right', fontsize=13) #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlim(-mgrid_size, mgrid_size)
plt.ylim(-mgrid_size, mgrid_size)
#plt.xlim(-1.215, -1.18)
#plt.ylim(-0.63, -0.62)
plt.tight_layout()
pulse_time_ranges_str = '_'.join([f'{pulse_time_range[0]}_{pulse_time_range[1]}' for pulse_time_range in pulse_time_ranges])
plt.savefig(f'figures/fitzhugh_nagumo_model_z_{z0[0]}_{z0[1]}_mu_{mu}_I_{pulse_time_ranges_str}.png', dpi=120)
plt.show()


# plot the voltage curve v(t):
fig, ax1 = plt.subplots(figsize=(6., 6))
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage $v(t)$', color=color)
ax1.plot(sol.t, sol.y[0], color=color, lw=2, label='Voltage $v(t)$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([-2.1, 2.0])
#ax1.set_yticks(np.arange(-2,2.0,1.0))
ax1.set_yticks(np.arange(-2.2,2.0,1.0))

# instantiate a second y-axis for I_ext(t):
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('$I_{ext}(t)$', color=color)
ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', )
#ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', drawstyle='steps')
#ax2.scatter(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$')
#ax2.plot(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$', marker='o', lw=0)
# instead of using plot, we can also use stem to visualize the delta pulse:
#markerline, stemlines, baseline = ax2.stem(sol.t, I_ext_t, linefmt='r-', markerfmt='ro', basefmt=' ')
#plt.setp(markerline, 'markersize', 4)  # change marker size
ax2.tick_params(axis='y', labelcolor=color)
# Remove box on the right and top for current axis
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(color)
ax2.set_ylim([-0.5, 2])
ax2.set_yticks(np.arange(0,2.0,0.5))

lines,  labels  = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.title(f'Voltage curve for pulsed current\na: {a}, b: {b}, $\mu$: {mu}, z0: {np.round(z0,2)}\n'+ \
          f"pulse intervals: {pulse_time_ranges}")
plt.tight_layout()
pulse_time_ranges_str = '_'.join([f'{pulse_time_range[0]}_{pulse_time_range[1]}' for pulse_time_range in pulse_time_ranges])
plt.savefig(f'figures/fitzhugh_nagumo_voltage_I_ext_pulsed_{z0[0]}_{z0[1]}_mu_{mu}_{pulse_time_ranges_str}.png', dpi=120)
plt.show()
# %% TIME-DEPENDENT EXTERNAL CURRENT: RAMPING-PLATEAUING CURRENT
# define the FitzHugh-Nagumo model with time-dependent ramping I_ext, now with
# adjustable time point of the onset of the ramping, the ramping having a plateau (not increasing anymore),
# adjustable time point when the plateau should be reached:
def fitzhugh_nagumo_time_dependent_ramping2(t, z, mu, a, b, I_ext_start, I_ext_end, eval_time, 
                                            ramp_start_time, plateau_time, t_iteration):
    v, w = z
    # calculate the time steps required to reach the plateau, i.e., into how many time steps the ramping
    # needs to be divided:
    dt = eval_time / t_iteration
    # within ramp_start_time to plateau_time, the ramping should be linearly increasing from I_ext_start to I_ext_end:
    if t < ramp_start_time:
        I_ext = 0
    elif t >= ramp_start_time and t < plateau_time:
        I_ext = I_ext_start + (I_ext_end - I_ext_start) * ((t - ramp_start_time) / (plateau_time - ramp_start_time))
    else:
        I_ext = I_ext_end
    
    dvdt = mu * (v - (v**3) / 3 - w + I_ext)
    dwdt = (1 / mu) * (v + a - b * w)
    return [dvdt, dwdt]

# set time span:
eval_time   = 50
t_iteration = 400
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

""" # define the x-array for the nullclines from -mgrid_size to mgrid_size with a spacing of t_iteration:
x_null = np.linspace(-mgrid_size, mgrid_size, t_iteration) """

# set parameters for I_ext pulsed:
I_ext = 0.3 # external current during pulse

# set parameters for I_ext ramping:
I_ext_start = 0.0  # Starting value of I_ext
I_ext_end = 1.0    # Ending value of I_ext
ramp_start_time = 10
plateau_time = 20

# set initial conditions:
#z0 = [1, 2]
z0 = [-1.2017543859649122, -0.6271929824561404]# same as before
mu = 2.0
a = 0.7
b = 0.8

# calculate the trajectory with time-dependent I_ext:
sol = solve_ivp(fitzhugh_nagumo_time_dependent_ramping2, t_span, z0, 
                args=(mu, a, b, I_ext_start, I_ext_end, eval_time, ramp_start_time, plateau_time, t_iteration), 
                t_eval=t_eval)

# calculate I_ext(t) for plotting:
I_ext_t = np.zeros_like(sol.t)
for time_i, time in enumerate(sol.t):
    if time < ramp_start_time:
        I_ext_t[time_i] = 0
    elif time >= ramp_start_time and time < plateau_time:
        I_ext_t[time_i] = I_ext_start + (I_ext_end - I_ext_start) * ((time - ramp_start_time) / (plateau_time - ramp_start_time))
    else:
        I_ext_t[time_i] = I_ext_end

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
# plot the streamline plot colored by the speed of the flow:
#speed = np.sqrt(u**2 + v**2)
#plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, v_nullcline(x_null, I_ext_start), '-', c="darkturquoise", lw=2, 
         label=f'v-nullcline for $I_{{ext}}={I_ext_start}$')
plt.plot(x_null, v_nullcline(x_null, I_ext_end), '--', c="darkturquoise", lw=2,
         label=f'v-nullcline for $I_{{ext}}={I_ext_end}$')
""" plt.plot(x_null, v_nullcline_time_dependent(sol.t, x_null, I_ext, pulse_time_ranges),
            '-', c="darkturquoise", lw=2) """
plt.plot(x_null, w_nullcline(x_null, a, b), '-', c="darkturquoise", lw=2)
plt.plot(sol.y[0], sol.y[1], 'r-', lw=3, label=f'Trajectory, $z_0$={np.round(z0,2)}')
# indicate start point:
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
# indicate the direction of the trajectory's last point with an arrow:
plt.title(f'phase plane plot: FitzHugh-Nagumo model\na: {a}, b: {b}, $\mu$: {mu},\n'+ \
          f"ramping interval: {ramp_start_time} to {plateau_time}")
plt.xlabel('v')
plt.ylabel('w')
plt.legend(loc='lower right', fontsize=13) #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlim(-mgrid_size, mgrid_size)
plt.ylim(-mgrid_size, mgrid_size)
#plt.xlim(-1.215, -1.18)
#plt.ylim(-0.63, -0.62)
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_model_phase_plane_ramped_{z0[0]}_{z0[1]}_mu_{mu}_{ramp_start_time}_{plateau_time}.png', dpi=120)
plt.show()


# plot the voltage curve v(t):
fig, ax1 = plt.subplots(figsize=(6., 6))
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage $v(t)$', color=color)
ax1.plot(sol.t, sol.y[0], color=color, lw=2, label='Voltage $v(t)$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([-2.1, 2.0])
ax1.set_yticks(np.arange(-2,2.0,1.0))

# instantiate a second y-axis for I_ext(t):
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('$I_{ext}(t)$', color=color)
ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', )
#ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', drawstyle='steps')
#ax2.scatter(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$')
#ax2.plot(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$', marker='o', lw=0)
# instead of using plot, we can also use stem to visualize the delta pulse:
#markerline, stemlines, baseline = ax2.stem(sol.t, I_ext_t, linefmt='r-', markerfmt='ro', basefmt=' ')
#plt.setp(markerline, 'markersize', 4)  # change marker size
ax2.tick_params(axis='y', labelcolor=color)
# Remove box on the right and top for current axis
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(color)
ax2.set_ylim([-0.5, 2])
ax2.set_yticks(np.arange(0,2.0,0.5))

lines,  labels  = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.title(f'Voltage curve for pulsed current\na: {a}, b: {b}, $\mu$: {mu}, z0: {np.round(z0,2)}\n'+ \
          f"ramping interval: [{ramp_start_time}, {plateau_time}]")
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_model_voltage_ramped_{z0[0]}_{z0[1]}_mu_{mu}_{ramp_start_time}_{plateau_time}.png', dpi=120)
plt.show()
# %% TIME-DEPENDENT EXTERNAL CURRENT: PLATEAUING-RAMPING CURRENT
# similar to the previous function, but now define it for a decreasing ramping:
def fitzhugh_nagumo_time_dependent_ramping_decreasing(t, z, mu, a, b, I_ext_start, I_ext_end, eval_time,
                                                        ramp_start_time, zero_plateau_time, t_iteration):
    v, w = z
    # calculate the time steps required to reach the plateau, i.e., into how many time steps the ramping
    # needs to be divided:
    dt = eval_time / t_iteration
    # within ramp_start_time to zero_plateau_time, the ramping should be linearly decreasing from I_ext_start to I_ext_end:
    if t < ramp_start_time:
        I_ext = I_ext_start
    elif t >= ramp_start_time and t < zero_plateau_time:
        I_ext = I_ext_start + (I_ext_end - I_ext_start) * (1 - ((t - ramp_start_time) / (zero_plateau_time - ramp_start_time)))
    else:
        I_ext = I_ext_end
    
    dvdt = mu * (v - (v**3) / 3 - w + I_ext)
    dwdt = (1 / mu) * (v + a - b * w)
    return [dvdt, dwdt]
    

# set time span:
eval_time   = 50
t_iteration = 400
t_span = [0, eval_time]
t_eval = np.linspace(*t_span, t_iteration)

""" # define the x-array for the nullclines from -mgrid_size to mgrid_size with a spacing of t_iteration:
x_null = np.linspace(-mgrid_size, mgrid_size, t_iteration) """

# set parameters for I_ext pulsed:
I_ext = 1.0 # external current during pulse

# set parameters for I_ext ramping:
I_ext_start = 1.0  # Starting value of I_ext
I_ext_end = 0.0    # Ending value of I_ext
ramp_start_time = 20
plateau_time = 30

# set initial conditions:
#z0 = [1, 2]
z0 = [-1.2017543859649122, -0.6271929824561404]# same as before
mu = 2.0
a = 0.7
b = 0.8

# calculate the trajectory with time-dependent I_ext:
sol = solve_ivp(fitzhugh_nagumo_time_dependent_ramping_decreasing, t_span, z0, 
                args=(mu, a, b, I_ext_start, I_ext_end, eval_time, ramp_start_time, plateau_time, t_iteration), 
                t_eval=t_eval)

# calculate I_ext(t) for plotting:
I_ext_t = np.zeros_like(sol.t)
for time_i, time in enumerate(sol.t):
    if time < ramp_start_time:
        I_ext_t[time_i] = I_ext_start
    elif time >= ramp_start_time and time < plateau_time:
        I_ext_t[time_i] = I_ext_start + (I_ext_end - I_ext_start) * ((time - ramp_start_time) / (plateau_time - ramp_start_time))
    else:
        I_ext_t[time_i] = I_ext_end

# print absolute minimum of the voltage curve:
print(f"Absolute minimum of the voltage curve: {np.min(sol.y[0])}")

# plot vector field and trajectory:
plt.figure(figsize=(6, 6))
plt.clf()
# plot the streamline plot colored by the speed of the flow:
#speed = np.sqrt(u**2 + v**2)
#plt.streamplot(x, y, u, v, color=speed, cmap='cool', density=2.0)
plt.plot(x_null, v_nullcline(x_null, I_ext_start), '-', c="darkturquoise", lw=2, 
         label=f'v-nullcline for $I_{{ext}}={I_ext_start}$')
plt.plot(x_null, v_nullcline(x_null, I_ext_end), '--', c="darkturquoise", lw=2,
         label=f'v-nullcline for $I_{{ext}}={I_ext_end}$')
""" plt.plot(x_null, v_nullcline_time_dependent(sol.t, x_null, I_ext, pulse_time_ranges),
            '-', c="darkturquoise", lw=2) """
plt.plot(x_null, w_nullcline(x_null, a, b), '-', c="darkturquoise", lw=2)
plt.plot(sol.y[0], sol.y[1], 'r-', lw=3, label=f'Trajectory, $z_0$={np.round(z0,2)}')
# indicate start point:
plt.plot(sol.y[0][0], sol.y[1][0], 'bo', label='start point', alpha=0.75, markersize=7)
plt.plot(sol.y[0][-1], sol.y[1][-1], 'o', c="yellow", label='end point', alpha=0.75, markersize=7)
# indicate the direction of the trajectory's last point with an arrow:
plt.title(f'phase plane plot: FitzHugh-Nagumo model\na: {a}, b: {b}, $\mu$: {mu},\n'+ \
          f"ramping interval: {ramp_start_time} to {plateau_time}")
plt.xlabel('v')
plt.ylabel('w')
plt.legend(loc='lower right', fontsize=13) #, bbox_to_anchor=(1, 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlim(-mgrid_size, mgrid_size)
plt.ylim(-mgrid_size, mgrid_size)
#plt.xlim(-1.215, -1.18)
#plt.ylim(-0.63, -0.62)
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_model_phase_plane_deramped_{z0[0]}_{z0[1]}_mu_{mu}_{ramp_start_time}_{plateau_time}.png', dpi=120)
plt.show()


# plot the voltage curve v(t):
fig, ax1 = plt.subplots(figsize=(6., 6))
color = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Voltage $v(t)$', color=color)
ax1.plot(sol.t, sol.y[0], color=color, lw=2, label='Voltage $v(t)$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([-2.1, 2.0])
#ax1.set_yticks(np.arange(-2,2.0,1.0))
ax1.set_yticks(np.arange(-2.2,2.0,1.0))

# instantiate a second y-axis for I_ext(t):
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('$I_{ext}(t)$', color=color)
ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', )
#ax2.plot(sol.t, I_ext_t, color=color, lw=2, linestyle='--', label='$I_{ext}(t)$', drawstyle='steps')
#ax2.scatter(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$')
#ax2.plot(sol.t, I_ext_t, color=color, label='$I_{ext}(t)$', marker='o', lw=0)
# instead of using plot, we can also use stem to visualize the delta pulse:
#markerline, stemlines, baseline = ax2.stem(sol.t, I_ext_t, linefmt='r-', markerfmt='ro', basefmt=' ')
#plt.setp(markerline, 'markersize', 4)  # change marker size
ax2.tick_params(axis='y', labelcolor=color)
# Remove box on the right and top for current axis
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(color)
ax2.set_ylim([-0.5, 2])
ax2.set_yticks(np.arange(0,2.0,0.5))

lines,  labels  = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.title(f'Voltage curve for pulsed current\na: {a}, b: {b}, $\mu$: {mu}, z0: {np.round(z0,2)}\n'+ \
          f"ramping interval: [{ramp_start_time}, {plateau_time}]")
plt.tight_layout()
plt.savefig(f'figures/fitzhugh_nagumo_model_voltage_deramped_{z0[0]}_{z0[1]}_mu_{mu}_{ramp_start_time}_{plateau_time}.png', dpi=120)
plt.show()
# %% END
