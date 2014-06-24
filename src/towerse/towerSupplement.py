#!/usr/bin/env python
# encoding: utf-8
"""
shellBuckling.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

from math import sqrt, cos, atan2, pi
import numpy as np
from commonse.utilities import CubicSplineSegment, cubic_spline_eval



def fatigue(M_DEL, N_DEL, d, t, m=4, DC=80.0, eta=1.265, stress_factor=1.0, weld_factor=True):
    """estimate fatigue damage for tower station

    Parmeters
    ---------
    M_DEL : array_like(float) (N*m)
        damage equivalent moment at tower section
    N_DEL : array_like(int)
        corresponding number of cycles in lifetime
    d : array_like(float) (m)
        tower diameter at section
    t : array_like(float) (m)
        tower shell thickness at section
    m : int
        slope of S/N curve
    DC : float (N/mm**2)
        some max stress from a standard
    eta : float
        safety factor
    stress_factor : float
        load_factor * stress_concentration_factor
    weld_factor : bool
        if True include an empirical weld factor

    Returns
    -------
    damage : float
        damage from Miner's rule for this tower section
    """


    # convert to mm
    dvec = np.array(d)*1e3
    tvec = np.array(t)*1e3

    nvec = len(d)
    damage = np.zeros(nvec)

    # initialize weld factor (added cubic spline around corner)
    if weld_factor:
        x1 = 24.0
        x2 = 26.0
        spline = CubicSplineSegment(x1, x2, 1.0, (25.0/x2)**0.25, 0.0, 25.0**0.25*-0.25*x2**-1.25)


    for i in range(nvec):

        d = dvec[i]
        t = tvec[i]

        # weld factor
        if not weld_factor or t <= x1:
            weld = 1.0
        elif t >= x2:
            weld = (25.0/t)**0.25
        else:
            weld = spline.eval(t)


        # stress
        r = d/2.0
        I = pi*r**3*t
        c = r
        sigma = M_DEL[i]*c/I * stress_factor * 1e3  # convert to N/mm^2

        # maximum allowed stress
        Smax = DC * weld / eta

        # number of cycles to failure
        Nf = (Smax/sigma)**m

        # number of cycles for this load
        N1 = 2e6  # TODO: where does this come from?
        N = N_DEL[i]/N1

        # damage
        damage[i] = N/Nf

    return damage


def vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress, gamma, sigma_y):
    """combine stress for von Mises"""

    # von mises stress
    a = ((axial_stress + hoop_stress)/2.0)**2
    b = ((axial_stress - hoop_stress)/2.0)**2
    c = shear_stress**2
    von_mises = np.sqrt(a + 3.0*(b+c))

    # stress margin
    stress_utilization = gamma * von_mises / sigma_y

    return stress_utilization



def hoopStressEurocode(windLoads, waveLoads, z, d, t, L_reinforced):
    """default method for computing hoop stress using Eurocode method"""

    r = d/2.0-t/2.0

    C_theta = 1.5
    omega = L_reinforced/np.sqrt(r*t)
    k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
    k_w = np.maximum(0.65, np.minimum(1.0, k_w))
    q_dyn = np.interp(z, windLoads.z, windLoads.q) + np.interp(z, waveLoads.z, waveLoads.q)
    Peq = k_w*q_dyn
    hoop_stress = -Peq*r/t

    return hoop_stress


def bucklingGL(d, t, Fz, Myy, tower_height, E, sigma_y, gamma_f=1.2, gamma_b=1.1, gamma_g=1.1):

    # other factors
    alpha = 0.21  # buckling imperfection factor
    beta = 1.0  # bending coefficient
    sk_factor = 2.0  # fixed-free
    tower_height = tower_height * sk_factor

    # geometry
    A = pi * d * t
    I = pi * (d/2.0)**3 * t
    Wp = I / (d/2.0)

    # applied loads
    Nd = -Fz * gamma_g
    Md = Myy * gamma_f

    # plastic resistance
    Np = A * sigma_y / gamma_b
    Mp = Wp * sigma_y / gamma_b

    # factors
    Ne = pi**2 * (E * I) / (1.1 * tower_height**2)
    lambda_bar = np.sqrt(Np * gamma_b / Ne)
    phi = 0.5 * (1 + alpha*(lambda_bar - 0.2) + lambda_bar**2)
    kappa = np.ones_like(d)
    idx = lambda_bar > 0.2
    kappa[idx] = 1.0 / (phi[idx] + np.sqrt(phi[idx]**2 - lambda_bar[idx]**2))
    delta_n = 0.25*kappa*lambda_bar**2
    delta_n = np.minimum(delta_n, 0.1)

    constraint = Nd/(kappa*Np) + beta*Md/Mp + delta_n

    return constraint





def shellBucklingEurocode(d, t, sigma_z, sigma_t, tau_zt, L_reinforced, E, sigma_y, gamma_f=1.2, gamma_b=1.1):
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

    n = len(d)
    constraint = np.zeros(n)

    for i in range(n):
        h = L_reinforced[i]

        r1 = d[i]/2.0 - t[i]/2.0
        r2 = d[i]/2.0 - t[i]/2.0
        t1 = t[i]
        t2 = t[i]

        sigma_z_shell = sigma_z[i]
        sigma_t_shell = sigma_t[i]
        tau_zt_shell = tau_zt[i]

        # TODO: the following is non-smooth, although in general its probably OK
        # change to magnitudes and add safety factor
        sigma_z_shell = gamma_f*abs(sigma_z_shell)
        sigma_t_shell = gamma_f*abs(sigma_t_shell)
        tau_zt_shell = gamma_f*abs(tau_zt_shell)

        constraint[i] = _shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z_shell, sigma_t_shell, tau_zt_shell, E[i], sigma_y[i])

    return constraint




def _cxsmooth(omega, rovert):

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
        Cx = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        Cx = 1.0

    elif omega >= ptL2 and omega <= ptR2:

        fL = 1.0
        fR = 1 + 0.2/Cxb*(1-2.0*ptR2/rovert)
        gL = 0.0
        gR = -0.4/Cxb/rovert
        Cx = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    elif omega > ptR2 and omega < ptL3:
        Cx = 1 + 0.2/Cxb*(1-2.0*omega/rovert)

    elif omega >= ptL3 and omega <= ptR3:

        fL = 1 + 0.2/Cxb*(1-2.0*ptL3/rovert)
        fR = 0.6
        gL = -0.4/Cxb/rovert
        gR = 0.0
        Cx = cubic_spline_eval(ptL3, ptR3, fL, fR, gL, gR, omega)

    else:
        Cx = 0.6

    return Cx


def _sigmasmooth(omega, E, rovert):

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

        sigma = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, omega)

    else:

        alpha1 = 0.92/1.63 - 2.03/1.63**4
        sigma = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/omega*rovert)**4)

    return sigma


def _tausmooth(omega, rovert):

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
        C_tau = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        C_tau = 1.0

    elif omega >= ptL2 and omega <= ptR2:
        fL = 1.0
        fR = 1.0/3.0*sqrt(ptR2/rovert) + 1 - sqrt(8.7)/3
        gL = 0.0
        gR = 1.0/6/sqrt(ptR2*rovert)
        C_tau = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    else:
        C_tau = 1.0/3.0*sqrt(omega/rovert) + 1 - sqrt(8.7)/3

    return C_tau



def _shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z, sigma_t, tau_zt, E, sigma_y):
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
    Cx = _cxsmooth(omega, rovert)


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
    alpha_z = 0.62/(1 + 1.91*(delta_wk/t)**1.44)

    chi_z = _buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z)

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

    sigma_t_Rcr = _sigmasmooth(omega, E, rovert)

    # buckling reduction factor
    alpha_t = 0.65  # high fabrication quality
    lambda_t0 = 0.4
    beta_t = 0.6
    eta_t = 1.0
    lambda_t = sqrt(sigma_y/sigma_t_Rcr)

    chi_theta = _buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t)

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
    C_tau = _tausmooth(omega, rovert)

    tau_zt_Rcr = 0.75*E*C_tau*sqrt(1.0/omega)/rovert

    # reduction factor
    alpha_tau = 0.65  # high fabrifaction quality
    beta_tau = 0.6
    lambda_tau0 = 0.4
    eta_tau = 1.0
    lambda_tau = sqrt(sigma_y/sqrt(3)/tau_zt_Rcr)

    chi_tau = _buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau)

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
        (tau_zt/tau_zt_Rd)**k_tau

    return buckling_constraint



def _buckling_reduction_factor(alpha, beta, eta, lambda_0, lambda_bar):
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

        chi = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, lambda_bar)

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
