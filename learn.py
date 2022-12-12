import pickle
import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt

import math
from scipy.ndimage import gaussian_filter1d
def derivative(x, t):
    dt = np.append(t[1:], np.nan) - t
    dx = np.append(x[1:], np.nan) - x
    xd = dx / dt

    # prev_max = max(xd)
#     xd = gaussian_filter1d(xd, 3)
    # xd = xd / (prev_max - max(xd))
    return xd
    


def get_f(x, t, alpha, beta):
    # get forcing function from demonstration
    # print(t)
    T = t[-1]
    q0 = x[0]
    g = x[-1]
    xd = derivative(x, t)
    xdd = derivative(xd, t)
    f = []
    K = alpha
    B = beta
    for q, qd, qdd in zip(x, xd, xdd):
        # qdd = K * (g - q) * (T**-2)  - B * qd * (T**-1) + (g - q0) * (T ** -2)
        f.append((qdd * (T ** 2) - K * (g - q) + B * qd * T) / (g - q0))
    f = np.nan_to_num(np.array(f))
    return f

# https://www.askpython.com/python/normal-distribution
def normal_dist(x, mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def learn_weights(traj, t, n_basis=31, alpha=15.0, beta=3.75, tau=0.125, C=None, H=None):
    if C is None:
        C = np.linspace(0, 1, n_basis)
    if H is None:
        H = [(0.65 * (1./(n_basis-1)) ** 2) for _ in range(len(C))]

    out = {
        'tau': tau,
        'alpha': alpha,
        'beta': beta,
        'num_dims': 3,
        'num_basis': n_basis,
        'num_sensors': 1,
        'phi_j': [1.0],
        'mu': list(C),
        'h': H,
    }
    # Phi = np.array([np.exp(-0.5 * ((t / max(t)) - c)**2 / h) for c, h in zip(C, H)])
    # todo this doesn't work
    # Phi = np.array([-h * np.exp((t / max(t) - c)**2) for c, h in zip(C, H)])
    phi = np.array([np.exp(-0.5 * ((t/max(t)) - c) ** 2 / h) for c, h in zip(C, H)])
    phi = phi / np.sum(phi, axis=0)
    weights = []
    for axis in range(0, 3):
        smoothed = gaussian_filter1d(traj[:, axis], 5)
        f_axis = get_f(smoothed, t, alpha, beta)
        w = (np.linalg.inv(phi @ phi.T) @ phi) @ f_axis
        weights.append(w)
    weights = np.stack(weights)
    weights = np.expand_dims(weights, axis=1)
    out["weights"] = weights
    return out

            

from scipy.integrate import odeint, ode
    
def get_pos_from_f_odeint(t, x0, g, C, H, weights, alpha=15, beta=3.75):
    T = t[-1]
    K = alpha
    B = beta
    q0 = x0



def get_pos_from_f(t, x0, g, C, H, weights, alpha=15, beta=3.75, solver="odeint"):
    T = t[-1]
    K = alpha
    B = beta
    
    q0 = x0



    if solver == "ode":
        def f(t, y):
            phi = np.array([math.exp(-0.5 * ((t/T) - c) ** 2 / h) for c, h in zip(C, H)])
            phi = phi / np.sum(phi, axis=0)
            f_t = np.dot(phi, weights)
            q = y[0]
            qd = y[1]
            return [qd, K * (g - q) * (T**-2) - B * qd * (T**-1) + (g - q0) * f_t * (T ** -2)]

        y0 = [q0, 0]
        t_range = t
        solver = ode(f)
        solver.set_integrator('dopri5')
        solver.set_initial_value(y0, t_range[0])
        y = []
        dt = 0.01
        new_t = []
        while solver.successful() and solver.t < T:
            new_t.append(solver.t+dt)
            y.append(solver.integrate(solver.t+dt))
        y = np.stack(y)[:,0]
        return y
    elif solver == "odeint":
        def f(y, t):
            phi = np.array([math.exp(-0.5 * ((t/T) - c) ** 2 / h) for c, h in zip(C, H)])
            phi = phi / np.sum(phi, axis=0)
            f_t = np.dot(phi, weights)
            q = y[0]
            qd = y[1]
            return [qd, K * (g - q) * (T**-2) - B * qd * (T**-1) + (g - q0) * f_t * (T ** -2)]
        y0 = [q0, 0]
        t_range = t

        y = odeint(f, y0, t_range)
        y = y[:,0]
        return y
    elif solver == "euler":
        T = t[-1]
        q0 = x0
        K = alpha
        B = beta
        q = q0
        qd = 0
        qdd = 0
        qdd_list = [qdd]
        qd_list = [qd]
        q_list = [q]
        prev_t = 0
        for t_i in t:        
            phi = np.array([math.exp(-0.5 * ((t_i/T) - c) ** 2 / h) for c, h in zip(C, H)])
            phi = phi / np.sum(phi, axis=0)
            f_t = np.dot(phi, weights)

            qdd = K * (g - q) * (T**-2) - B * qd * (T**-1) + (g - q0) * f_t * (T ** -2)
            qd = qd +  qdd * (t_i - prev_t)
            q = q + qd * (t_i - prev_t)
            prev_t = t_i
            qdd_list.append(qdd)
            qd_list.append(qd)
            q_list.append(q)
        return q_list

if __name__ == "__main__":
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        raw_traj = pickle.load(f)
    trajectory = np.stack(raw_traj["pose"])
    # print(np.array(raw_traj["t"]) - raw_traj['t'][0])
    t = np.stack(raw_traj["t"])
    t = t - t[0]
    # print(t)
    # t = np.arange(0, 10, 0.1)
    out = learn_weights(trajectory, t, n_basis=5, alpha=25*25/4, beta=25)
    # print(out["weights"] @ )
    # print(out["weights"])
    print(out)

    
