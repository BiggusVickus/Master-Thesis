import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE for multiple populations: dy_i/dt = -k_i * y_i
def g(R, v, K):
    return (v*R) / (K + R)

def ode(t, y, e, v, K, r, M, tau, beta, washin, washout):
    R, U, I1, I2, I3, I4, P = y
    dRdt = -e * g(R, v, K) * (U + (I1 + I2 + I3 + I4)) - washout * R + washin
    dUdt = g(R, v, K) * U - r * U * P - washout * U
    dI1 = r * U * P - M/tau*I1 - washout*I1
    dI2 = M/tau*I1 - M/tau*I2 - washout*I2
    dI3 = M/tau*I2 - M/tau*I3 - washout*I3
    dI4 = M/tau*I3 - M/tau*I4 - washout*I4
    dP = beta * M/tau * I4 - r *(U + I1 + I2 + I3 + I4) * P - washout * P
    return [dRdt, dUdt, dI1, dI2, dI3, dI4, dP]

# Parameters
e=0.14375872112626925
v=1.3813047017599906
K=134.84822010949117
r=0.001
M=4
tau=2.1466321025719455
beta=49
washin=2
washout=0

y0 = np.array([395, 50, 0, 0, 0, 0, 10])
t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)

# Solve the ODE
sol = solve_ivp(ode, t_span, y0, args=(e, v, K, r, M, tau, beta, washin, washout), t_eval=t_eval)

for i in range(len(y0)):
    print(f"Population {i+1} at final time: {sol.y[i][-1]:.8f}")
    
# for i in range(len(y0)-1):
#     plt.plot(sol.t, sol.y[i], label=f'Population {i+1}')

derivative = np.gradient(sol.y[0], sol.t)
for i in range(1, len(sol.y[0])):
    print(sol.y[0][i])
    print(sol.t[i])
    print('---')
print(derivative)

# plt.plot(sol.t, derivative, linestyle='--', label=f'Population {i+1}')
plt.plot(sol.t, sol.y[1], linestyle='--', label=f'Uninfected')
plt.plot(sol.t, sol.y[2]+sol.y[3]+sol.y[4]+sol.y[5], linestyle='--', label=f'Infected')
plt.plot(sol.t, sol.y[1] + sol.y[2]+sol.y[3]+sol.y[4]+sol.y[5], linestyle='--', label=f'Infected')

plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Multiple Populations: dy/dt = -k*y')
plt.legend()
plt.grid()
plt.show()