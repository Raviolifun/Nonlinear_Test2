import math
import numpy as np
import random

class Dynamics:
    def __init__(self, mx, mphi, cx, cphi, l, ax, aphi, xdmax, fxd, phidmax, fphid, alpha=1, beta=1, delta=1):
        self.mx = mx
        self.mphi = mphi
        self.cx = cx
        self.cphi = cphi
        self.l = l
        self.ax = ax
        self.aphi = aphi

        self.xdmax = xdmax
        self.fxd = fxd

        self.phidmax = phidmax
        self.fphid = fphid

        self.th_x = np.transpose(np.array([[self.mx, self.mphi, self.cx, self.mphi * l]]))
        self.th_phi = np.transpose(np.array([[self.mphi * l**2, self.cphi, self.mphi * l]]))

        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.big_gamma_inv_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.big_gamma_x = np.linalg.inv(self.big_gamma_inv_x)
        self.lil_gamma_x = 1

        self.sumyxU = np.array([[0, 0, 0, 0]])
        self.sumyxyx = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.KTheta_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.KCL_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.sumyaxU = 0
        self.sumyaxyax = 0
        self.KTheta_ax = 1,
        self.KCL_ax = 1

        self.big_gamma_inv_phi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.big_gamma_phi = np.linalg.inv(self.big_gamma_inv_phi)
        self.lil_gamma_phi = 1

        self.sumyphiU = np.array([[0, 0, 0]])
        self.sumyphiyphi = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.KTheta_phi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.KCL_phi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.sumyaphiU = 0
        self.sumyaphiyaphi = 0
        self.KTheta_aphi = 1,
        self.KCL_aphi = 1

        self.g = 9.8

    def f(self, t, y):
        x1, x2, ux, phi1, phi2, uphi, thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4, ah1_x, thh1_phi_1, thh1_phi_2, thh1_phi_3, ah1_phi = y

        thh1_x = np.transpose(np.array([[thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4]]))
        thh1_phi = np.transpose(np.array([[thh1_phi_1, thh1_phi_2, thh1_phi_3]]))

        # ==================================================================================
        # ==============================      Dynamics      ================================
        # ==================================================================================

        dx1dt = x2
        dx2dt = 1/(self.mx + self.mphi * math.sin(phi1)**2) * (-self.cx * x2 + self.mphi * self.g * math.cos(phi1)
                                                * math.sin(phi1) + self.mphi * self.l * phi2**2 * math.sin(phi1) + ux)
        x3 = dx2dt

        dphi1dt = phi2
        dphi2dt = 1/(self.mphi * self.l**2) * (- self.cphi * phi2 - self.mphi * self.l * self.g * math.sin(phi1)
                                               - self.mphi * self.l * math.cos(phi1) * x3**2 + uphi)
        phi3 = dphi2dt

        # ==================================================================================
        # ==============================     Error time     ================================
        # ==================================================================================

        xd_1 = self.xdmax * math.sin(2 * math.pi * self.fxd * t)
        phid_1 = self.phidmax * math.sin(2 * math.pi * self.fphid * t)
        xd_2 = 2 * math.pi * self.fxd * self.xdmax * math.cos(2 * math.pi * self.fxd * t)
        phid_2 = 2 * math.pi * self.fphid * self.phidmax * math.cos(2 * math.pi * self.fphid * t)
        xd_3 = -(2 * math.pi * self.fxd * self.xdmax)**2 * math.sin(2 * math.pi * self.fxd * t)
        phid_3 = -(2 * math.pi * self.fphid * self.phidmax)**2 * math.sin(2 * math.pi * self.fphid * t)
        xd_4 = -(2 * math.pi * self.fxd * self.xdmax)**3 * math.cos(2 * math.pi * self.fxd * t)
        phid_4 = -(2 * math.pi * self.fphid * self.phidmax)**3 * math.cos(2 * math.pi * self.fphid * t)

        e1_x = xd_1 - x1
        e2_x = xd_2 - x2
        e3_x = xd_3 - x3
        e1_phi = phid_1 - phi1
        e2_phi = phid_2 - phi2
        e3_phi = phid_3 - phi3

        r1_x = e2_x + self.alpha * e1_x
        r2_x = e3_x + self.alpha * e2_x
        r1_phi = e2_phi + self.alpha * e1_phi
        r2_phi = e3_phi + self.alpha * e2_phi

        Y1_x = np.array([[xd_3 + self.alpha * e2_x, (math.sin(phi1)**2) * (xd_3 + self.alpha * e2_x) + math.cos(phi1)
                          * math.sin(phi1) * (phi2 * r1_x - self.g), x2, -(phi2**2) * math.sin(phi1)]])
        Y2_x = np.array([[xd_4 + self.alpha * e3_x,
                          2 * math.sin(phi1) * math.cos(phi1) * phi2 * (xd_3 + self.alpha * e2_x) +
                          math.sin(phi1)**2 * (xd_4 - self.alpha * e3_x) -
                          math.sin(phi1) * phi2 * math.sin(phi1) * (phi2 * r1_x - self.g) +
                          math.cos(phi1) * (math.cos(phi1) * phi2 * (phi2 * r1_x - self.g) + math.sin(phi1) * (phi3 * r1_x + phi2 * r2_x)),
                          x3,
                          -2 * phi2 * phi3 * math.sin(phi1) - (phi2**2) * math.cos(phi1)]])

        yx = np.array([[x3, math.sin(phi1) ** 2 * xd_3 - self.g * math.cos(phi1) * math.sin(phi1), x2, -phi2 ** 2 * math.sin(phi1)]])
        yphi = np.array([[phi3, phi2, self.g * math.sin(phi1) + math.cos(phi1) * x3]])
        yax = ux
        yaphi = uphi

        Ux = ux
        Uphi = uphi

        thh2_x = np.matmul(self.big_gamma_x, np.transpose(Y1_x)) * r1_x + np.matmul(self.big_gamma_x * self.KTheta_x, (np.transpose(Y1_x) * Ux - np.matmul(np.transpose(Y1_x) * Y1_x, thh1_x)))
        + np.matmul(self.big_gamma_x * self.KCL_x, (self.sumyxU - np.matmul(self.sumyxyx, thh1_x)))

        [thh2_x_1, thh2_x_2, thh2_x_3, thh2_x_4] = thh2_x

        ud1_x = np.matmul(Y1_x, thh1_x) + e1_x + self.beta * r1_x
        ud2_x = np.matmul(Y2_x, thh1_x) + np.matmul(Y1_x, thh2_x) + e2_x + self.beta * r2_x

        u_tilde_x = ud1_x - ux

        mu_x = ud2_x + ah1_x * ux + r1_x + self.delta * u_tilde_x

        [[duxdt]] = - self.ax * ux + mu_x

        x4 = 1/(self.mx + self.mphi * math.sin(phi1)**2) * (-(2 * self.mphi * math.sin(phi1) * math.cos(phi1) * phi2)
                                                            * x3 - self.cx * x3 + self.mphi * self.g *
                                                            (math.cos(phi1)**2 - math.sin(phi1)**2) * phi2 + self.mphi *
                                                            self.l * (2 * phi2 * phi3 * math.sin(phi1)
                                                                      + phi2**2 * math.cos(phi1)) + duxdt)

        Y1_phi = np.array([[phid_3 + self.alpha * e2_phi, phi2, self.g * math.sin(phi1) + math.cos(phi1) * x3]])
        Y2_phi = np.array([[phid_4 + self.alpha * e3_phi, phi3, self.g * math.cos(phi1) * phi2 - math.sin(phi1) * x3 + math.cos(phi1) * x4]])

        thh2_phi = np.matmul(self.big_gamma_phi, np.transpose(Y1_phi)) * r1_phi + np.matmul(self.big_gamma_phi * self.KTheta_phi, (np.transpose(Y1_phi) * Uphi - np.matmul(np.transpose(Y1_phi) * Y1_phi, thh1_phi)))
        + np.matmul(self.big_gamma_phi * self.KCL_phi, (self.sumyphiU - np.matmul(self.sumyphiyphi, thh1_phi)))
        [thh2_phi_1, thh2_phi_2, thh2_phi_3] = thh2_phi

        ud1_phi = np.matmul(Y1_phi, thh1_phi) + e1_phi + self.beta * r1_phi
        ud2_phi = np.matmul(Y2_phi, thh1_phi) + np.matmul(Y1_phi, thh2_phi) + e2_phi + self.beta * r2_phi

        u_tilde_phi = ud1_phi - uphi


        mu_phi = ud2_phi + ah1_phi * uphi + r1_phi + self.delta * u_tilde_phi

        [[duphidt]] = - self.aphi * uphi + mu_phi

        Uax = mu_x - ud2_x
        Uaphi = mu_phi - ud2_phi

        ah2_x = self.lil_gamma_x * u_tilde_x * ux + self.lil_gamma_x * self.KTheta_ax * (yax*Uax - yax * yax * ah1_x)
        + self.lil_gamma_x * self.KCL_ax * (self.sumyaxU - self.sumyaxyax * ah1_x)

        ah2_phi = self.lil_gamma_phi * u_tilde_phi * uphi + self.lil_gamma_phi * self.KTheta_aphi * (yaphi*Uaphi - yaphi * yaphi * ah1_phi)
        + self.lil_gamma_phi * self.KCL_aphi * (self.sumyaphiU - self.sumyaphiyaphi * ah1_phi)



        min_eig = 0.0001
        if random.random() < 1/10:
            if np.min(np.linalg.eigvals(self.sumyxyx)) < min_eig:
                self.sumyxU = self.sumyxU + np.transpose(yx) * Ux
                self.sumyxyx = self.sumyxyx + np.matmul(np.transpose(yx), yx)

            if np.min(np.linalg.eigvals(self.sumyphiyphi)) < min_eig:
                self.sumyphiU = self.sumyphiU + np.transpose(yphi) * Uphi
                self.sumyphiyphi = self.sumyphiyphi + np.matmul(np.transpose(yphi), yphi)

            if self.sumyaxyax < min_eig:
                self.sumyaxU = self.sumyaxU + yax * Uax
                self.sumyaxyax = self.sumyaxyax + yax**2

            if self.sumyaphiyaphi < min_eig:
                self.sumyaphiU = self.sumyaphiU + yaphi * Uaphi
                self.sumyaphiyaphi = self.sumyaphiyaphi + yaphi**2

        return [dx1dt, dx2dt, duxdt, dphi1dt, dphi2dt, duphidt, thh2_x_1, thh2_x_2, thh2_x_3, thh2_x_4, ah2_x,
                thh2_phi_1, thh2_phi_2, thh2_phi_3, ah2_phi]