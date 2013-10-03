#!/usr/bin/env python
# encoding: utf-8
"""
shellBuckling.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

from math import sqrt, cos, atan2
import numpy as np


def shellBuckling(self, npt, sigma_z, sigma_t, tau_zt, L_reinforced, gamma_f=1.2, gamma_b=1.1):
    """
    Estimate shell buckling constraint along tower.

    Arguments:
    npt - number of locations at each node at which stress is evaluated.
    sigma_z - axial stress at npt*node locations.  must be in order
                  [(node1_pts1-npt), (node2_pts1-npt), ...]
    sigma_t - azimuthal stress given at npt*node locations
    tau_zt - shear stress (z, theta) at npt*node locations
    E - modulus of elasticity
    sigma_y - yield stress
    L_reinforced - reinforcement length - structure is re-discretized with this spacing
    gamma_f - safety factor for stresses
    gamma_b - safety factor for buckling

    Returns:
    z
    an array of shell buckling constraints evaluted at (z[0] at npt locations,
    z[0]+L_reinforced at npt locations, ...).
    Each constraint must be <= 0 to avoid failure.
    """

    # break up into chunks of length L_reinforced
    z_re = np.arange(self.z[0], self.z[-1], L_reinforced)
    if (z_re[-1] != self.z[-1]):
        z_re = np.r_[z_re, self.z[-1]]

    # initialize
    constraint = np.zeros(npt * (len(z_re) - 1))

    # evaluate each line separately
    for j in range(npt):

        # pull off stresses along line
        sigma_z_line = sigma_z[j::npt]
        sigma_t_line = sigma_t[j::npt]
        tau_zt_line = tau_zt[j::npt]

        # interpolate into sections
        d_re = np.interp(z_re, self.z, self.d)
        t_re = np.interp(z_re, self.z, self.t)
        sigma_z_re = np.interp(z_re, self.z, sigma_z_line)
        sigma_t_re = np.interp(z_re, self.z, sigma_t_line)
        tau_zt_re = np.interp(z_re, self.z, tau_zt_line)

        for i in range(len(z_re)-1):
            h = z_re[i+1] - z_re[i]
            r1 = d_re[i] / 2.0
            r2 = d_re[i+1] / 2.0
            t1 = t_re[i]
            t2 = t_re[i+1]
            sigma_z_shell = sigma_z_re[i]  # use base value - should be conservative
            sigma_t_shell = sigma_t_re[i]
            tau_zt_shell = tau_zt_re[i]

            # only compressive stresses matter.
            # also change to magnitudes and add safety factor
            sigma_z_shell = gamma_f*abs(min(sigma_z_shell, 0.0))
            sigma_t_shell = gamma_f*abs(sigma_t_shell)
            tau_zt_shell = gamma_f*abs(tau_zt_shell)

            constraint[i*4 + j] = self.__shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z_shell, sigma_t_shell, tau_zt_shell)

    return z_re[0:-1], constraint



def __cubicspline(self, ptL, ptR, fL, fR, gL, gR, pts):

    A = np.array([[ptL**3, ptL**2, ptL, 1],
                  [ptR**3, ptR**2, ptR, 1],
                  [3*ptL**2, 2*ptL, 1, 0],
                  [3*ptR**2, 2*ptR, 1, 0]])
    b = np.array([fL, fR, gL, gR])

    coeff = np.linalg.solve(A, b)

    value = coeff[0]*pts**3 + coeff[1]*pts**2 + coeff[2]*pts + coeff[3]

    return value


def __cxsmooth(self, omega, rovert):

    Cxb = 6.0  # clamped-clamped
    constant = 1 + 1.83/1.7 - 2.07/1.7**2

    ptL1 = 1.7-0.25
    ptR1 = 1.7+0.25

    ptL2 = 0.5*rovert - 1.0
    ptR2 = 0.5*rovert + 1.0

    ptL3 = (0.5+Cxb)*rovert - 1.0
    ptR3 = (0.5+Cxb)*rovert + 1.0


    if omega < ptL1:
        Cx = constant - 1.83/omega + 2.07/omega**2

    elif omega >= ptL1 and omega <= ptR1:

        fL = constant - 1.83/ptL1 + 2.07/ptL1**2
        fR = 1.0
        gL = 1.83/ptL1**2 - 4.14/ptL1**3
        gR = 0.0
        Cx = self.__cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        Cx = 1.0

    elif omega >= ptL2 and omega <= ptR2:

        fL = 1.0
        fR = 1 + 0.2/Cxb*(1-2.0*ptR2/rovert)
        gL = 0.0
        gR = -0.4/Cxb/rovert
        Cx = self.__cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

    elif omega > ptR2 and omega < ptL3:
        Cx = 1 + 0.2/Cxb*(1-2.0*omega/rovert)

    elif omega >= ptL3 and omega <= ptR3:

        fL = 1 + 0.2/Cxb*(1-2.0*ptL3/rovert)
        fR = 0.6
        gL = -0.4/Cxb/rovert
        gR = 0.0
        Cx = self.__cubicspline(ptL3, ptR3, fL, fR, gL, gR, omega)

    else:
        Cx = 0.6

    return Cx


def __sigmasmooth(self, omega, E, rovert):

    Ctheta = 1.5  # clamped-clamped

    ptL = 1.63*rovert*Ctheta - 1
    ptR = 1.63*rovert*Ctheta + 1

    if omega < 20.0*Ctheta:
        offset = (10.0/(20*Ctheta)**2 - 5/(20*Ctheta)**3)
        Cthetas = 1.5 + 10.0/omega**2 - 5/omega**3 - offset
        sigma = 0.92*E*Cthetas/omega/rovert

    elif omega >= 20.0*Ctheta and omega < ptL:

        sigma = 0.92*E*Ctheta/omega/rovert

    elif omega >= ptL and omega <= ptR:

        alpha1 = 0.92/1.63 - 2.03/1.63**4

        fL = 0.92*E*Ctheta/ptL/rovert
        fR = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/ptR*rovert)**4)
        gL = -0.92*E*Ctheta/rovert/ptL**2
        gR = -E*(1.0/rovert)*2.03*4*(Ctheta/ptR*rovert)**3*Ctheta/ptR**2

        sigma = self.__cubicspline(ptL, ptR, fL, fR, gL, gR, omega)

    else:

        alpha1 = 0.92/1.63 - 2.03/1.63**4
        sigma = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/omega*rovert)**4)

    return sigma


def __tausmooth(self, omega, rovert):

    ptL1 = 9
    ptR1 = 11

    ptL2 = 8.7*rovert - 1
    ptR2 = 8.7*rovert + 1

    if omega < ptL1:
        C_tau = sqrt(1.0 + 42.0/omega**3 - 42.0/10**3)

    elif omega >= ptL1 and omega <= ptR1:
        fL = sqrt(1.0 + 42.0/ptL1**3 - 42.0/10**3)
        fR = 1.0
        gL = -63.0/ptL1**4/fL
        gR = 0.0
        C_tau = self.__cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        C_tau = 1.0

    elif omega >= ptL2 and omega <= ptR2:
        fL = 1.0
        fR = 1.0/3.0*sqrt(ptR2/rovert) + 1 - sqrt(8.7)/3
        gL = 0.0
        gR = 1.0/6/sqrt(ptR2*rovert)
        C_tau = self.__cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

    else:
        C_tau = 1.0/3.0*sqrt(omega/rovert) + 1 - sqrt(8.7)/3

    return C_tau



def __shellBucklingOneSection(self, h, r1, r2, t1, t2, gamma_b, sigma_z, sigma_t, tau_zt):
    """
    Estimate shell buckling for one tapered cylindrical shell section.

    Arguments:
    h - height of conical section
    r1 - radius at bottom
    r2 - radius at top
    t1 - shell thickness at bottom
    t2 - shell thickness at top
    E - modulus of elasticity
    sigma_y - yield stress
    gamma_b - buckling reduction safety factor
    sigma_z - axial stress component
    sigma_t - azimuthal stress component
    tau_zt - shear stress component (z, theta)

    Returns:
    buckling_constraint, which must be <= 0 to avoid failure

    """

    E = self.E
    sigma_y = self.sigma_y

    #NOTE: definition of r1, r2 switched from Eurocode document to be consistent with FEM.

    # ----- geometric parameters --------
    beta = atan2(r1-r2, h)
    L = h/cos(beta)
    t = 0.5*(t1+t2)

    # ------------- axial stress -------------
    # length parameter
    le = L
    re = 0.5*(r1+r2)/cos(beta)
    omega = le/sqrt(re*t)
    rovert = re/t

    # compute Cx
    Cx = self.__cxsmooth(omega, rovert)


    # if omega <= 1.7:
    #     Cx = 1.36 - 1.83/omega + 2.07/omega/omega
    # elif omega > 0.5*rovert:
    #     Cxb = 6.0  # clamped-clamped
    #     Cx = max(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))
    # else:
    #     Cx = 1.0

    # critical axial buckling stress
    sigma_z_Rcr = 0.605*E*Cx/rovert

    # compute buckling reduction factors
    lambda_z0 = 0.2
    beta_z = 0.6
    eta_z = 1.0
    Q = 25.0  # quality parameter - high
    lambda_z = sqrt(sigma_y/sigma_z_Rcr)
    delta_wk = 1.0/Q*sqrt(rovert)*t
    alpha_z = 0.62/(1 + 1.91*delta_wk/t)**1.44

    chi_z = self.__buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z)

    # design buckling stress
    sigma_z_Rk = chi_z*sigma_y
    sigma_z_Rd = sigma_z_Rk/gamma_b

    # ---------------- hoop stress ------------------

    # length parameter
    le = L
    re = 0.5*(r1+r2)/(cos(beta))
    omega = le/sqrt(re*t)
    rovert = re/t

    # Ctheta = 1.5  # clamped-clamped
    # CthetaS = 1.5 + 10.0/omega**2 - 5.0/omega**3

    # # critical hoop buckling stress
    # if (omega/Ctheta < 20.0):
    #     sigma_t_Rcr = 0.92*E*CthetaS/omega/rovert
    # elif (omega/Ctheta > 1.63*rovert):
    #     sigma_t_Rcr = E*(1.0/rovert)**2*(0.275 + 2.03*(Ctheta/omega*rovert)**4)
    # else:
    #     sigma_t_Rcr = 0.92*E*Ctheta/omega/rovert

    sigma_t_Rcr = self.__sigmasmooth(omega, E, rovert)

    # buckling reduction factor
    alpha_t = 0.65  # high fabrication quality
    lambda_t0 = 0.4
    beta_t = 0.6
    eta_t = 1.0
    lambda_t = sqrt(sigma_y/sigma_t_Rcr)

    chi_theta = self.__buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t)

    sigma_t_Rk = chi_theta*sigma_y
    sigma_t_Rd = sigma_t_Rk/gamma_b

    # ----------------- shear stress ----------------------

    # length parameter
    le = h
    rho = sqrt((r1+r2)/(2.0*r2))
    re = (1.0 + rho - 1.0/rho)*r2*cos(beta)
    omega = le/sqrt(re*t)
    rovert = re/t

    # if (omega < 10):
    #     C_tau = sqrt(1.0 + 42.0/omega**3)
    # elif (omega > 8.7*rovert):
    #     C_tau = 1.0/3.0*sqrt(omega/rovert)
    # else:
    #     C_tau = 1.0
    C_tau = self.__tausmooth(omega, rovert)

    tau_zt_Rcr = 0.75*E*C_tau*sqrt(1.0/omega)/rovert

    # reduction factor
    alpha_tau = 0.65  # high fabrifaction quality
    beta_tau = 0.6
    lambda_tau0 = 0.4
    eta_tau = 1.0
    lambda_tau = sqrt(sigma_y/sqrt(3)/tau_zt_Rcr)

    chi_tau = self.__buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau)

    tau_zt_Rk = chi_tau*sigma_y/sqrt(3)
    tau_zt_Rd = tau_zt_Rk/gamma_b

    # buckling interaction parameters

    k_z = 1.25 + 0.75*chi_z
    k_theta = 1.25 + 0.75*chi_theta
    k_tau = 1.75 + 0.25*chi_tau
    k_i = (chi_z*chi_theta)**2

    # buckling constraint

    buckling_constraint = \
        (sigma_z/sigma_z_Rd)**k_z + \
        (sigma_t/sigma_t_Rd)**k_theta - \
        k_i*(sigma_z*sigma_t/sigma_z_Rd/sigma_t_Rd) + \
        (tau_zt/tau_zt_Rd)**k_tau - 1

    return buckling_constraint



def __buckling_reduction_factor(self, alpha, beta, eta, lambda_0, lambda_bar):
    """
    Computes a buckling reduction factor used in Eurocode shell buckling formula.
    """

    lambda_p = sqrt(alpha/(1.0-beta))

    ptL = 0.9*lambda_0
    ptR = 1.1*lambda_0

    if (lambda_bar < ptL):
        chi = 1.0

    elif lambda_bar >= ptL and lambda_bar <= ptR:  # cubic spline section

        fracR = (ptR-lambda_0)/(lambda_p-lambda_0)
        fL = 1.0
        fR = 1-beta*fracR**eta
        gL = 0.0
        gR = -beta*eta*fracR**(eta-1)/(lambda_p-lambda_0)

        chi = self.__cubicspline(ptL, ptR, fL, fR, gL, gR, lambda_bar)

    elif lambda_bar > ptR and lambda_bar < lambda_p:
        chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

    else:
        chi = alpha/lambda_bar**2



    # if (lambda_bar <= lambda_0):
    #     chi = 1.0
    # elif (lambda_bar >= lambda_p):
    #     chi = alpha/lambda_bar**2
    # else:
    #     chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

    return chi
