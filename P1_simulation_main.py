import P1_dynamics
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import math


if __name__ == '__main__':
    save = True  # Whether it saves the graphs

    tspan = np.linspace(0, 40, 1000)
    number = 100

    def make_plot(name, state):
        plt.figure(name)
        plt.clf()
        plt.title(state)
        plt.ylabel(state)
        plt.xlabel("$t$ $(sec)$")
        plt.grid()

    name_state_dict = {
        "feedback_errors": "feedback_errors(t)",
        "estimation_errors": "estimation_errors(t)",
        "total_input": "total_input(t)",
        "feedback_error": "feedback_error(t)",
        "feedforward_input": "feedforward_input(t)"
    }

    for key in name_state_dict:
        make_plot(key, name_state_dict[key])

    save_total_input       = [None] * number
    save_feedback_errors   = [None] * number
    save_feedforward_input = [None] * number

    for i in range(number):
        print(i)
        # mx, mphi, cx, cphi, l, ax, aphi, xdmax, fxd, phidmax, fphid
        dynamics = P1_dynamics.Dynamics(random.random()/5 + 0.2,  # mx
                                        random.random()/5 + 0.2,  # mphi
                                        random.random()/5 + 5,  # cx
                                        random.random()/5 + 5,  # cphi
                                        random.random()/20 + 0.2,  # l
                                        random.random()/5 + 1,  # ax
                                        random.random()/5 + 1,  # aphi
                                        random.random()/10 + 0.1,  # xdmax
                                        random.random()/10 + 0.1,  # fxd
                                        random.random()/10 + 0.1,  # phidmax
                                        random.random()/10 + 0.1)  # fphid

        # x1, x2, ux, phi1, phi2, uphi, thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4,
        # ah1_x, thh1_phi_1, thh1_phi_2, thh1_phi_3, ah1_phi
        yinit = [(random.random() - 0.5) * 1 * 2,  # x1
                 0,  # x2
                 0,  # ux
                 (random.random() - 0.5) * 0.4 * math.pi,  # phi1
                 0,  # phi2
                 0,  # uphi
                 random.random() * 2 + 0.1,  # thh1_x_1
                 random.random() * 2 + 0.1,  # thh1_x_2
                 random.random() * 2 + 0.1,  # thh1_x_3
                 random.random() * 2 + 0.1,  # thh1_x_4
                 random.random() * 2 + 1,  # ah1_x
                 random.random() * 2 + 0.1,  # thh1_phi_1
                 random.random() * 2 + 0.1,  # thh1_phi_2
                 random.random() * 2 + 0.1,  # thh1_phi_3
                 random.random() * 2 + 1]  # ah1_phi

        # Solve differential equation
        # sol = solve_ivp(lambda t, y: dynamics.f(t, y),
        #                 [tspan[0], tspan[-1]], yinit, t_eval=tspan, rtol=1e-5)
        sol = solve_ivp(lambda t, y: dynamics.f(t, y),
                        [tspan[0], tspan[-1]], yinit, t_eval=tspan)

        if sol.status < 0:
            print(sol.message)

        time = sol.t
        [x1, x2, ux, phi1, phi2, uphi, thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4, ah1_x, thh1_phi_1,
         thh1_phi_2, thh1_phi_3, ah1_phi] = sol.y

        def func_dyn(t, x1, x2, ux, phi1, phi2, uphi, thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4, ah1_x, thh1_phi_1,
         thh1_phi_2, thh1_phi_3, ah1_phi):

            th1_x = np.transpose(np.array([[dynamics.mx, dynamics.mphi, dynamics.cx, dynamics.mphi * dynamics.l]]))
            th1_phi = np.transpose(np.array([[dynamics.mphi * dynamics.l**2, dynamics.cphi, dynamics.mphi * dynamics.l]]))

            thh1_x = np.transpose(np.array([[thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4]]))
            thh1_phi = np.transpose(np.array([[thh1_phi_1, thh1_phi_2, thh1_phi_3]]))

            # ==================================================================================
            # ==============================      Dynamics      ================================
            # ==================================================================================

            dx1dt = x2
            dx2dt = 1 / (dynamics.mx + dynamics.mphi * math.sin(phi1) ** 2) * (
                        -dynamics.cx * x2 + dynamics.mphi * dynamics.g * math.cos(phi1)
                        * math.sin(phi1) + dynamics.mphi * dynamics.l * phi2 ** 2 * math.sin(phi1) + ux)
            x3 = dx2dt

            dphi1dt = phi2
            dphi2dt = 1 / (dynamics.mphi * dynamics.l ** 2) * (- dynamics.cphi * phi2 - dynamics.mphi * dynamics.l * dynamics.g * math.sin(phi1)
                                                       - dynamics.mphi * dynamics.l * math.cos(phi1) * x3 ** 2 + uphi)
            phi3 = dphi2dt

            # ==================================================================================
            # ==============================     Error time     ================================
            # ==================================================================================

            xd_1 = dynamics.xdmax * math.sin(2 * math.pi * dynamics.fxd * t)
            phid_1 = dynamics.phidmax * math.sin(2 * math.pi * dynamics.fphid * t)
            xd_2 = 2 * math.pi * dynamics.fxd * dynamics.xdmax * math.cos(2 * math.pi * dynamics.fxd * t)
            phid_2 = 2 * math.pi * dynamics.fphid * dynamics.phidmax * math.cos(2 * math.pi * dynamics.fphid * t)
            xd_3 = -(2 * math.pi * dynamics.fxd * dynamics.xdmax) ** 2 * math.sin(2 * math.pi * dynamics.fxd * t)
            phid_3 = -(2 * math.pi * dynamics.fphid * dynamics.phidmax) ** 2 * math.sin(2 * math.pi * dynamics.fphid * t)
            xd_4 = -(2 * math.pi * dynamics.fxd * dynamics.xdmax) ** 3 * math.cos(2 * math.pi * dynamics.fxd * t)
            phid_4 = -(2 * math.pi * dynamics.fphid * dynamics.phidmax) ** 3 * math.cos(2 * math.pi * dynamics.fphid * t)

            e1_x = xd_1 - x1
            e2_x = xd_2 - x2
            e3_x = xd_3 - x3
            e1_phi = phid_1 - phi1
            e2_phi = phid_2 - phi2
            e3_phi = phid_3 - phi3

            r1_x = e2_x + dynamics.alpha * e1_x
            r2_x = e3_x + dynamics.alpha * e2_x
            r1_phi = e2_phi + dynamics.alpha * e1_phi
            r2_phi = e3_phi + dynamics.alpha * e2_phi

            Y1_x = np.array(
                [[xd_3 + dynamics.alpha * e2_x, (math.sin(phi1) ** 2) * (xd_3 + dynamics.alpha * e2_x) + math.cos(phi1)
                  * math.sin(phi1) * (phi2 * r1_x - dynamics.g), x2, -(phi2 ** 2) * math.sin(phi1)]])
            Y2_x = np.array([[xd_4 + dynamics.alpha * e3_x,
                              2 * math.sin(phi1) * math.cos(phi1) * phi2 * (xd_3 + dynamics.alpha * e2_x) +
                              math.sin(phi1) ** 2 * (xd_4 - dynamics.alpha * e3_x) -
                              math.sin(phi1) * phi2 * math.sin(phi1) * (phi2 * r1_x - dynamics.g) +
                              math.cos(phi1) * (math.cos(phi1) * phi2 * (phi2 * r1_x - dynamics.g) + math.sin(phi1) * (
                                          phi3 * r1_x + phi2 * r2_x)),
                              x3,
                              -2 * phi2 * phi3 * math.sin(phi1) - (phi2 ** 2) * math.cos(phi1)]])

            yx = np.array([[x3, math.sin(phi1) ** 2 * xd_3 - dynamics.g * math.cos(phi1) * math.sin(phi1), x2,
                            -phi2 ** 2 * math.sin(phi1)]])
            yphi = np.array([[phi3, phi2, dynamics.g * math.sin(phi1) + math.cos(phi1) * x3]])
            yax = ux
            yaphi = uphi

            Ux = ux
            Uphi = uphi

            thh2_x = np.matmul(dynamics.big_gamma_x, np.transpose(Y1_x)) * r1_x + np.matmul(
                dynamics.big_gamma_x * dynamics.KTheta_x,
                (np.transpose(Y1_x) * Ux - np.matmul(np.transpose(Y1_x) * Y1_x, thh1_x)))
            + np.matmul(dynamics.big_gamma_x * dynamics.KCL_x, (dynamics.sumyxU - np.matmul(dynamics.sumyxyx, thh1_x)))

            [thh2_x_1, thh2_x_2, thh2_x_3, thh2_x_4] = thh2_x

            ud1_x = np.matmul(Y1_x, thh1_x) + e1_x + dynamics.beta * r1_x
            ud2_x = np.matmul(Y2_x, thh1_x) + np.matmul(Y1_x, thh2_x) + e2_x + dynamics.beta * r2_x

            u_tilde_x = ud1_x - ux

            mu_x = ud2_x + ah1_x * ux + r1_x + dynamics.delta * u_tilde_x

            [[duxdt]] = - dynamics.ax * ux + mu_x

            x4 = 1 / (dynamics.mx + dynamics.mphi * math.sin(phi1) ** 2) * (
                        -(2 * dynamics.mphi * math.sin(phi1) * math.cos(phi1) * phi2)
                        * x3 - dynamics.cx * x3 + dynamics.mphi * dynamics.g *
                        (math.cos(phi1) ** 2 - math.sin(phi1) ** 2) * phi2 + dynamics.mphi *
                        dynamics.l * (2 * phi2 * phi3 * math.sin(phi1)
                                  + phi2 ** 2 * math.cos(phi1)) + duxdt)

            Y1_phi = np.array([[phid_3 + dynamics.alpha * e2_phi, phi2, dynamics.g * math.sin(phi1) + math.cos(phi1) * x3]])
            Y2_phi = np.array([[phid_4 + dynamics.alpha * e3_phi, phi3,
                                dynamics.g * math.cos(phi1) * phi2 - math.sin(phi1) * x3 + math.cos(phi1) * x4]])

            thh2_phi = np.matmul(dynamics.big_gamma_phi, np.transpose(Y1_phi)) * r1_phi + np.matmul(
                dynamics.big_gamma_phi * dynamics.KTheta_phi,
                (np.transpose(Y1_phi) * Uphi - np.matmul(np.transpose(Y1_phi) * Y1_phi, thh1_phi)))
            + np.matmul(dynamics.big_gamma_phi * dynamics.KCL_phi, (dynamics.sumyphiU - np.matmul(dynamics.sumyphiyphi, thh1_phi)))
            [thh2_phi_1, thh2_phi_2, thh2_phi_3] = thh2_phi

            ud1_phi = np.matmul(Y1_phi, thh1_phi) + e1_phi + dynamics.beta * r1_phi
            ud2_phi = np.matmul(Y2_phi, thh1_phi) + np.matmul(Y1_phi, thh2_phi) + e2_phi + dynamics.beta * r2_phi

            u_tilde_phi = ud1_phi - uphi

            mu_phi = ud2_phi + ah1_phi * uphi + r1_phi + dynamics.delta * u_tilde_phi

            [[duphidt]] = - dynamics.aphi * uphi + mu_phi

            Uax = mu_x - ud2_x
            Uaphi = mu_phi - ud2_phi

            ah2_x = dynamics.lil_gamma_x * u_tilde_x * ux + dynamics.lil_gamma_x * dynamics.KTheta_ax * (
                        yax * Uax - yax * yax * ah1_x)
            + dynamics.lil_gamma_x * dynamics.KCL_ax * (dynamics.sumyaxU - dynamics.sumyaxyax * ah1_x)

            ah2_phi = dynamics.lil_gamma_phi * u_tilde_phi * uphi + dynamics.lil_gamma_phi * dynamics.KTheta_aphi * (
                        yaphi * Uaphi - yaphi * yaphi * ah1_phi)
            + dynamics.lil_gamma_phi * dynamics.KCL_aphi * (dynamics.sumyaphiU - dynamics.sumyaphiyaphi * ah1_phi)

            min_eig = 0.0001
            if random.random() < 1 / 10:
                if np.min(np.linalg.eigvals(dynamics.sumyxyx)) < min_eig:
                    dynamics.sumyxU = dynamics.sumyxU + np.transpose(yx) * Ux
                    dynamics.sumyxyx = dynamics.sumyxyx + np.matmul(np.transpose(yx), yx)

                if np.min(np.linalg.eigvals(dynamics.sumyphiyphi)) < min_eig:
                    dynamics.sumyphiU = dynamics.sumyphiU + np.transpose(yphi) * Uphi
                    dynamics.sumyphiyphi = dynamics.sumyphiyphi + np.matmul(np.transpose(yphi), yphi)

                if dynamics.sumyaxyax < min_eig:
                    dynamics.sumyaxU = dynamics.sumyaxU + yax * Uax
                    dynamics.sumyaxyax = dynamics.sumyaxyax + yax ** 2

                if dynamics.sumyaphiyaphi < min_eig:
                    dynamics.sumyaphiU = dynamics.sumyaphiU + yaphi * Uaphi
                    dynamics.sumyaphiyaphi = dynamics.sumyaphiyaphi + yaphi ** 2

            tht1_x = th1_x - thh1_x
            tht1_phi = th1_phi - thh1_phi
            at_x = dynamics.ax - ah1_x
            at_phi = dynamics.aphi - ah1_phi

            feedback_errors = (e1_x**2 + e1_phi**2 + r1_x**2 + r1_phi**2)**0.5
            estimation_errors = (np.linalg.norm(tht1_x)**2 + np.linalg.norm(tht1_phi)**2 + at_x**2 + at_phi**2)**0.5

            total_input = (ux**2 + uphi**2)**0.5
            feedback_input = ((e1_x + dynamics.beta * r1_x)**2 + (e1_phi + dynamics.beta * r1_phi)**2)**0.5
            feedforward_input = (np.linalg.norm(np.matmul(Y1_x, thh1_x))**2 + np.linalg.norm(np.matmul(Y1_phi, thh1_phi))**2)**0.5

            return feedback_errors, estimation_errors, total_input, feedback_input, feedforward_input

        func_dyn_vector = np.vectorize(func_dyn)
        feedback_errors, estimation_errors, total_input, feedback_input, feedforward_input = func_dyn_vector(time,
                x1, x2, ux, phi1, phi2, uphi, thh1_x_1, thh1_x_2, thh1_x_3, thh1_x_4, ah1_x, thh1_phi_1,
                thh1_phi_2, thh1_phi_3, ah1_phi)

        plt.figure("feedback_errors")
        plt.plot(time, feedback_errors)

        plt.figure("estimation_errors")
        plt.plot(time, estimation_errors)

        plt.figure("total_input")
        plt.plot(time, total_input)

        plt.figure("feedback_error")
        plt.plot(time, feedback_errors)

        plt.figure("feedforward_input")
        plt.plot(time, feedforward_input)

        save_total_input[i]       = total_input
        save_feedback_errors[i]   = feedback_errors
        save_feedforward_input[i] = feedforward_input

    for key in name_state_dict:
        plt.figure(key)
        if save:
            plt.savefig('Saves/' + key)
        else:
            plt.show()

    dif1 = [None]*number
    dif2 = [None]*number

    for i in range(number):
        dif1[i] = save_total_input[i] - save_feedback_errors[i]
        dif2[i] = save_total_input[i] - save_feedforward_input[i]

    sumdif1 = 0
    sumdif2 = 0

    for i in range(number):
        sumdif1 = sumdif1 + dif1[i]
        sumdif2 = sumdif2 + dif2[i]

    avg1 = sumdif1 / number
    avg2 = sumdif2 / number

    make_plot("total_input_minus_feedback_error_inputs", "total_input_minus_feedback_error_inputs(t)")
    plt.plot(time, avg1)
    plt.savefig('Saves/' + "total_input_minus_feedback_error_inputs(t)")

    make_plot("total_input_minus_feedforward_error_inputs", "total_input_minus_feedforward_error_inputs(t)")
    plt.plot(time, avg2)
    plt.savefig('Saves/' + "total_input_minus_feedforward_error_inputs(t)")

