import numpy as np
import matplotlib.pyplot as plt

# Params
L = 1


# Space grid size N
            # ∆x = L/N
N1 = 10  # For ∆x = 1/10
N2 = 100  # For ∆x = 1/100

# ∆x 
dx1 = L / N1
dx2 = L / N2

# Determining dt from the stability criterion
dt1 = 0.5 * dx1**2  
dt2 = 0.5 * dx2**2  

T = 1  # Total time

# Temporal grid size  M
M1 = int(T / dt1)
M2 = int(T / dt2)

# x axis
x1 = np.linspace(0, L, N1)
x2 = np.linspace(0, L, N2)


# Initialize solution matrices
u1 = np.zeros((N1, M1+1))
u2 = np.zeros((N2, M2+1))

# Set initial condition
u1[:, 0] = np.sin(np.pi * x1)
u2[:, 0] = np.sin(np.pi * x2)

# Set boundary conditions
u1[0, :] = 0
u2[0, :] = 0

u1[-1, :] = 0
u2[-1, :] = 0


def explicit_scheme(u, dx, dt, N, M):
    """
    Function which performs the explicit scheme.
    
    """
    alpha = dt / (dx ** 2)
    if alpha > 0.5:  # Stability condition
        raise ValueError(f"Stability condition not met. alpha = {alpha}, for dt={dt} and dx^2 ={dx**2}")
    
    print(f'N = {N}')
    print(f'M = {M}')
    for j in range(0, M-1):    # time 
        for i in range(1, N-1):    # space 
            u[i, j+1] = alpha*u[i-1, j] + (1-2*alpha)*u[i,j] + alpha*u[i+1, j]

    return u



# Perform the explicit scheme
u1 = explicit_scheme(u1, dx1, dt1, N1, M1+1)
u2 = explicit_scheme(u2, dx2, dt2, N2, M2+1)


# Studying the solutions at two time points
t1 = 0.01  # time point t1
t2 = 0.2   # time point t2

# Time index for u1
t1_index1 = int(t1/dt1)
t2_index1 = int(t2/dt1)

# Time index for u2
t1_index2 = int(t1/dt2)
t2_index2 = int(t2/dt2)

def analytical(x, t, D):
    a = np.sin(np.pi*x)*np.exp(-D*np.pi**2*t)
    return a


plt.figure()
plt.plot(x1, u1[:, t1_index1], 'blue', label=f"t1 = {t1_index1 * dt1:.2f}")
plt.plot(x1, u1[:, t2_index1], 'red', label=f"t2 = {t2_index1 * dt1:.2f}")
plt.plot(x1, analytical(x1, t1, 1), 'c.', label=f"Analytical t1 = {t1_index1 * dt1:.2f}")
plt.plot(x1, analytical(x1, t2, 1), 'k-.', label=f"Analytical t2 = {t2_index1 * dt1:.2f}")
plt.title("Numerical Solution for ∆x = 1/10")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.savefig('Explicit_scheme_1_10.pdf')

plt.figure()
plt.plot(x2, u2[:, t1_index2], 'blue', label=f"t1 = {t1_index2 * dt2:.2f}")
plt.plot(x2, u2[:, t2_index2], 'red', label=f"t2 = {t2_index2 * dt2:.2f}")
plt.plot(x2, analytical(x2, t1, 1), 'c.', label=f"Analytical t1 = {t1_index2 * dt2:.2f}")
plt.plot(x2, analytical(x2, t2, 1), 'k-.', label=f"Analytical t2 = {t2_index2 * dt2:.2f}")
plt.title("Numerical Solution for ∆x = 1/100")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.savefig('Explicit_scheme_1_100.pdf')


''' For a diffusion coefficient D not equal 1'''
"""
Need to get FTCS euler for a diffusion coefficient D
"""