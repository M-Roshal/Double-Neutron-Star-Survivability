"""
this codes simulates the survival of a DNS system for elliptical and circular cases. 
Elliptical case of particular interest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple

from astropy import constants as const
from astropy import units as u
from scipy.integrate import solve_ivp

from scipy.stats import rayleigh, ks_2samp, wasserstein_distance


G_AU: float = const.G.to_value(u.AU**3 / (u.Msun * u.day**2)) # G in AU^3/(Msun*day^2)
AU_PER_DAY_KMS: float = (const.au / u.day).to(u.km / u.s).value # AU per day to km/s
RNG = np.random.default_rng(seed=42) # random number generator with a fixed seed for reproducibility
np.random.seed(42)  # Ensure reproducibility for np.random-based functions (which we used in mass functions)

G = const.G.to_value(u.m**3 / (u.kg * u.s**2)) # G in mks
c = const.c.to_value(u.m / u.s) # c in mks
pi = np.pi


def post_sn_orbit_e(
    m1_pre: np.ndarray,          # mass of the primary star before SN {SM}
    m1_post: np.ndarray,         # mass of the primary star after SN {SM}
    m2: np.ndarray,              # mass of the companion star {SM}
    a0: np.ndarray,              # array of initial semi-major axes (not distances between bodies) {AU}
    v_kick_vec: np.ndarray,      # kick velocity vectors {km/s}
    f_ini: np.ndarray,           # true anomaly sampling (from low to high, uniform and random)
    e_ini: np.ndarray            # eccentricity sampling (from 0 to 1, uniform and random)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function simulates the post-supernova semi-major axis and eccentricity of a binary system. This is the elliptical case.
    """
    # pre-defined astrodynamics parameters

    M0 = m1_pre + m2 # initial total mass of a reduced two-body system {M_sun}
    Mt = m1_post + m2 # total mass of a reduced two-body system after SN {M_sun}
    v_c = np.sqrt(G_AU * M0 / a0) * AU_PER_DAY_KMS # reference circular orbital speed {km/s} (for a0, not used for elliptical velocity)

    r_mag = a0 * (1 - e_ini**2) / (1 + e_ini * np.cos(f_ini)) # instantaneous separation at explosion (ellipse) {AU}
    r_vec = np.column_stack((r_mag, np.zeros_like(r_mag), np.zeros_like(r_mag))) # position vector at explosion (x-axis, elliptical case) {AU}

    # Elliptical orbital velocity at explosion (true anomaly f_ini)
    v_orb_x = -v_c * np.sin(f_ini) / np.sqrt(1 - e_ini**2) # x-component (ellipse)
    v_orb_y = v_c * (np.cos(f_ini) + e_ini) / np.sqrt(1 - e_ini**2) # y-component (ellipse)
    v_orb_z = np.zeros_like(v_orb_x) # z-component (ellipse, always zero in the orbital plane)
    v_orb_vec = np.column_stack((v_orb_x, v_orb_y, v_orb_z)) # orbital velocity vector at explosion (ellipse) {km/s}
    v_orb_mag = np.sqrt(G_AU * M0 * (2/r_mag - 1/a0)) * AU_PER_DAY_KMS # orbital speed magnitude at explosion (Vis-Viva, ellipse)

    # Astrodynamics calculations for elliptical orbits
    v_fin_vec = v_orb_vec + v_kick_vec # final velocity vector after SN (ellipse)
    v_fin_mag2 = np.sum(v_fin_vec**2, axis=1) # final speed squared {km^2/s^2}
    v_fin_mag2_AU = v_fin_mag2 / AU_PER_DAY_KMS**2 # convert to AU^2/day^2

    mu_post = G_AU * Mt # gravitational parameter after SN {AU^3/day^2}
    specific_E = 0.5 * v_fin_mag2_AU - mu_post / r_mag  # specific orbital energy at explosion (ellipse) {AU^2/day^2}
    bound = specific_E < 0.0 # bound if specific energy is negative (ellipse)

    # Bound case calculations (ellipse)
    a_final = np.full_like(a0, np.nan) # final semi-major axis (ellipse)
    e_final = np.full_like(a0, np.nan) # final eccentricity (ellipse)

    if np.any(bound): # process only bound systems
        a_final[bound] = -mu_post[bound] / (2.0 * specific_E[bound]) # semi-major axis post SN (ellipse)
        v_fin_AU = v_fin_vec[bound] / AU_PER_DAY_KMS # final velocity {AU/day}
        h_vec = np.cross(r_vec[bound], v_fin_AU) # specific angular momentum vector {AU^2/day}
        h2 = np.sum(h_vec**2, axis=1) # squared magnitude of specific angular momentum vector {AU^4/day^2}
        e_sq = 1.0 - h2 / (mu_post[bound] * a_final[bound]) # eccentricity squared (ellipse)
        e_sq[e_sq < 0.0] = 0.0 # remove negative eccentricities
        e_final[bound] = np.sqrt(e_sq) # final eccentricity for bound systems (ellipse)

    return bound, a_final, e_final # return bound status, final semi-major axis, and final eccentricity (initial case - ellipse)

def post_sn_orbit_c(
    m1_pre: np.ndarray,          # mass of the primary star before SN
    m1_post: np.ndarray,         # mass of the primary star after SN
    m2: np.ndarray,              # mass of the companion star
    a0: np.ndarray,         # array of initial semi-major axes
    v_kick_vec: np.ndarray, # kick velocity vectors {km/s}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is the circular case of DNS survival simulation function.
    """
    v_orb = np.sqrt(G_AU * (m1_pre + m2) / a0) * AU_PER_DAY_KMS # relative orbital speed pre-explosion {km/s}. Here, a0 is the array of initial semi-major axes.
    v_orb_vec = np.zeros((a0.size, 3)) # creates a zero array for the orbital velocity vectors
    v_orb_vec[:, 1] = v_orb # sets the vector components for the orbital velocity. It sets the y-component to the orbital speed from line above.
    
    v_fin_vec = v_orb_vec + v_kick_vec # final velocity vector (via vector addition)
    v_fin_mag2 = np.sum(v_fin_vec**2, axis=1) # final speed squared {km^2/s^2}
    
    v_fin_mag2_AU = v_fin_mag2 / AU_PER_DAY_KMS**2 # convert to AU^2/day^2
    mu_post = G_AU * (m1_post + m2) # gravitational parameter after SN {AU^3/day^2} --- this is now an array ---
    specific_E = 0.5 * v_fin_mag2_AU - mu_post / a0 # specific orbital energy {AU^2/day^2}
    bound = specific_E < 0.0 # if bound, then specific energy is negative (array of boolean values)
    
    a_final = np.full_like(a0, np.nan) # create the same array shape as a0, filled with NaN for a_final (final semi-major axis)
    e_final = np.full_like(a0, np.nan) # create the same array shape as a0, filled with NaN for e_final (final eccentricity)

    if np.any(bound): # processes only if bound
        a_final[bound] = -mu_post[bound] / (2.0 * specific_E[bound]) # semi-major axis post SN {AU}
        r_vec = np.zeros((bound.sum(), 3)) # position vector for the bound system for the star that rotates (also the distance between the two stars)
        r_vec[:, 0] = a0[bound]             # sets the position vector x-component as the initial semi-major axis for the bound stars {AU} (Explosion is instantaneous)
        v_fin_AU = v_fin_vec[bound] / AU_PER_DAY_KMS # final velocity
        h_vec = np.cross(r_vec, v_fin_AU) # specific angular momentum vector via cross product of position and velocity vectors {AU^2/day}
        h2 = np.sum(h_vec**2, axis=1) # squared magnitude of specific angular momentum vector {AU^4/day^2}
        e_sq = 1.0 - h2 / (mu_post[bound] * a_final[bound]) # eccentricity squared
        e_sq[e_sq < 0.0] = 0.0 # remove eccentricities that are negative
        e_final[bound] = np.sqrt(e_sq) # final eccentricity for bound systems

    return bound, a_final, e_final # return the bound status, final semi-major axis vector, and final eccentricity

if __name__ == "__main__":

    # example params
    m1_pre = np.array([7.0, 3.0, 5.0])
    m1_post = np.array([4.33, 1.38, 3.43])
    m2 = np.array([6.35, 7.33, 7.33])
    a0 = np.array([1.0, 5.0, 10.0])
    v_kick_vec = np.array([[10.0, 0.0, 0.0], [5.0, 2.0, 10.0], [5.0, 5.0, 0.0]])


    f_ini = np.array([0.0, 0.0, 0.707]) # first two set to zero to compare computation in both cases
    e_ini = np.array([0.0, 0.0, 0.34])

    e_circ, a_circ, e_final_circ = post_sn_orbit_c(m1_pre, m1_post, m2, a0, v_kick_vec)
    e_ellip, a_ellip, e_final_ellip = post_sn_orbit_e(m1_pre, m1_post, m2, a0, v_kick_vec, f_ini, e_ini)

    print("Circular Initial Orbits:")
    for i in range(len(m1_pre)):
        print(f"System {i+1}: Bound = {e_circ[i]}, a_final = {a_circ[i]:.3f} AU, e_final = {e_final_circ[i]:.3f}")

    print("\nElliptical Initial Orbits:")
    for i in range(len(m1_pre)):
        print(f"System {i+1}: Bound = {e_ellip[i]}, a_final = {a_ellip[i]:.3f} AU, e_final = {e_final_ellip[i]:.3f}")


    print('\n\nNotice the first two results shall be the same, and the last one differes due to e and f0!')