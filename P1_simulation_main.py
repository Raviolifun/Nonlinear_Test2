import P1_dynamics
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import math


if __name__ == '__main__':
    save = False  # Whether it saves the graphs

    tspan = np.linspace(0, 40, 10000)
    number = 10

    def make_plot(name, state):
        plt.figure(name)
        plt.clf()
        plt.title(state)
        plt.ylabel(state)
        plt.xlabel("$t$ $(sec)$")
        plt.grid()

    name_state_dict = {
        "error": "e(t)",
        "errorTrack": "r(t)",
        "uTilde": "u tilde(t)",
        "mu": "mu(t)",
        "thetaTilde": "theta tilde(t)",
        "aTilde": "a tilde(t)"
    }

    for key in name_state_dict:
        make_plot(key, name_state_dict[key])

    for i in range(number):
        xd = (random.random() - 0.5) * 2 * 10
        print(xd)
        dynamics = P1_dynamics.Dynamics(random.random() * 5 + 0.5,
                                        random.random() * 5 + 0.5,
                                        random.random() * 5 + 0.5,
                                        random.random() * 5 + 0.5,
                                        random.random() * 5 + 0.5,
                                        9.81,
                                        xd,  # xd
                                        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                        1,
                                        random.random() * 2 + 1,
                                        random.random() * 2 + 1,
                                        random.random() * 2 + 1)
        yinit = [(random.random() - 0.5) * 2 * 100, 0, 0,
                 (random.random() - 0.5) * 2 * 100,
                 (random.random() - 0.5) * 2 * 100,
                 (random.random() - 0.5) * 2 * 100,
                 (random.random() - 0.5) * 2 * 100]
        # Solve differential equation
        sol = solve_ivp(lambda t, y: dynamics.f(t, y),
                        [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol=1e-5)

        time = sol.t
        [x1, x2, u, thh1_1, thh1_2, thh1_3, ah1] = sol.y

        def func_dyn(x1, x2, u, thh1_1, thh1_2, thh1_3, ah1):
            thh1 = np.transpose(np.array([[thh1_1, thh1_2, thh1_3]]))
            tht = dynamics.th - thh1

            dx2dt = 1 / dynamics.m * (-dynamics.c * x2 - dynamics.k * x1 + dynamics.m * dynamics.g * math.sin(dynamics.phi) + u)

            x3 = dx2dt

            e1 = dynamics.xd - x1
            e2 = -x2
            e3 = -x3
            r1 = e2 + dynamics.alpha * e1
            r2 = e3 + dynamics.alpha * e2

            Y1 = np.array([[dynamics.alpha * e2 - dynamics.g * math.sin(dynamics.phi), x2, x1]])
            Y2 = np.array([[dynamics.alpha * e3, x3, x2]])

            thh2 = np.matmul(dynamics.big_gamma, np.transpose(Y1)) * r1

            ud1 = np.matmul(Y1, thh1) + e1 + dynamics.beta * r1
            ud2 = np.matmul(Y2, thh1) + np.matmul(Y1, thh2) + e2 + dynamics.beta * r2

            u_tilde = ud1 - u
            ah2 = dynamics.lil_gamma * u_tilde * u

            mu = ud2 + ah1 * u + r1 + dynamics.delta * u_tilde

            ahh = (dynamics.a - ah2)

            return tht[0], tht[1], tht[2], e1, r1, u_tilde, mu, ahh


        func_dyn_vector = np.vectorize(func_dyn)
        tht1, tht2, tht3, e1, r1, u_tilde, mu, ahh = func_dyn_vector(x1, x2, u, thh1_1, thh1_2, thh1_3, ah1)

        plt.figure("error")
        plt.plot(time, e1)

        plt.figure("errorTrack")
        plt.plot(time, r1)

        plt.figure("uTilde")
        plt.plot(time, u_tilde)

        plt.figure("mu")
        plt.plot(time, mu)

        plt.figure("thetaTilde")
        plt.plot(time, tht1)
        plt.plot(time, tht2)
        plt.plot(time, tht3)

        plt.figure("aTilde")
        plt.plot(time, ahh)

    for key in name_state_dict:
        plt.figure(key)
        if save:
            plt.savefig('Saves/et')
        else:
            plt.show()

