import math
import numpy as np


class Dynamics:
    def __init__(self, m, c, k, a, phi, g, xd, big_gamma_inv, lil_gamma, alpha=0.3, beta=0.3, delta=0.3):
        self.m = m
        self.c = c
        self.k = k
        self.th = np.transpose(np.array([[self.m, self.c, self.k]]))
        self.g = g
        self.a = a
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.big_gamma_inv = big_gamma_inv
        self.big_gamma = np.linalg.inv(big_gamma_inv)
        self.lil_gamma = lil_gamma

        self.xd = xd

    def f(self, t, y):
        # Gonna have to reconstruct all the other things... not sure how. grab what I can from y!
        x1, x2, u, thh1_1, thh1_2, thh1_3, ah1 = y
        thh1 = np.transpose(np.array([[thh1_1, thh1_2, thh1_3]]))

        dx1dt = x2
        dx2dt = 1/self.m * (-self.c * x2 - self.k * x1 + self.m * self.g * math.sin(self.phi) + u)

        x3 = dx2dt

        e1 = self.xd - x1
        e2 = -x2
        e3 = -x3
        r1 = e2 + self.alpha * e1
        r2 = e3 + self.alpha * e2

        Y1 = np.array([[self.alpha * e2 - self.g * math.sin(self.phi), x2, x1]])
        Y2 = np.array([[self.alpha * e3, x3, x2]])

        thh2 = np.matmul(self.big_gamma, np.transpose(Y1)) * r1
        thh2_1, thh2_2, thh2_3 = thh2

        ud1 = np.matmul(Y1, thh1) + e1 + self.beta * r1
        ud2 = np.matmul(Y2, thh1) + np.matmul(Y1, thh2) + e2 + self.beta * r2

        u_tilde = ud1 - u
        ah2 = self.lil_gamma * u_tilde * u

        mu = ud2 + ah1 * u + r1 + self.delta * u_tilde

        dudt = - self.a * u + mu
        return [dx1dt, dx2dt, dudt, thh2_1, thh2_2, thh2_3, ah2]