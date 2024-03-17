# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from numpy import real
# set global font size for plots:
plt.rcParams.update({'font.size': 14})
# %% EIGENVECTORS AND NODES
# define systems:
A_saddle   = np.array([[2, 0], [0, -2]])  # saddle point
A_instable = np.array([[2, 0], [0, 2]])   # unstable node
A_stable   = np.array([[-2, 0], [0, -2]]) # stable knot

# calculate eigenvalues and eigenvectors:
eigvals_saddle, eigvecs_saddle = eig(A_saddle)
eigvals_instable, eigvecs_instable = eig(A_instable)
eigvals_stable, eigvecs_stable = eig(A_stable)

# plot the phase portraits for each system:
def plot_phase_portrait(A, eigvecs, eigvals, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    x, y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
    u = A[0, 0] * x + A[0, 1] * y
    v = A[1, 0] * x + A[1, 1] * y
    ax.streamplot(x, y, u, v, color='b')
    for vec in eigvecs.T:
        # indicate the direction of the eigenvectors and consider the sign of the real part:
        ax.quiver(0, 0, 3*real(vec[0]), 3*real(vec[1]), scale=1, 
                  scale_units='xy', angles='xy', color='r')
    ax.set_title(title+f"\neigenvalues (real part): {real(eigvals)}")
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'figures/eigenvectors_and_nodes_{title}.png', dpi=120)
    plt.show()

# saddle point:
plot_phase_portrait(A_saddle, eigvecs_saddle, eigvals_saddle, 'saddle point')

# unstable node:
plot_phase_portrait(A_instable, eigvecs_instable, eigvals_instable, 'unstable node')

# stable knot:
plot_phase_portrait(A_stable, eigvecs_stable, eigvals_stable, 'stable knot')



# %%
