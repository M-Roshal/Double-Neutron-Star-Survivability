"""kick_survival.py

University of Texas at Austin
Freshman Research Initiative, "White Dwarfs" Stream

Research Instructor: Dr. Michael Montgomery
Undergraduate researchers: Sid Chunduri, Alejandro Gonzalez, Cyril John, Michael Roshal, Nancy Devane
===================================================
DNS survival simulation

This program models whether a binary star system can survive a supernova explosion, 
if such an explosion imparts a "kick" velocity vector to the newborn neutron star and removes some mass.

INSTRUCTIONS:
---------------------------------------------------
1.  To use this script, you must first install the necessary packages. See them below.
2.  The script will load the DNS catalogue of the selected type from a local cache file.
    It must either be present in the same directory as this script, or you can specify a different path in the main loop.
3.  You must also specify the type of the companion star in the DNS system. See the main loop for the options.
"""

# libraries and dependencies --(required)--

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple

from astropy import constants as const
from astropy import units as u
from scipy.integrate import solve_ivp

from scipy.stats import rayleigh, ks_2samp, wasserstein_distance


# libraries and dependencies --(optional)--
# Without these, plotting will be unavailable.

# Set plotting libraries to None by default (import libraries to see plots)
plt = None
sns = None
Axes3D = None

# Try to import matplotlib.pyplot
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Try to import seaborn
try:
    import seaborn as sns
except ImportError:
    sns = None

# Try to import Axes3D for 3D plotting
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plots
except ImportError:
    Axes3D = None

# -------------------------------------------------------------------------------
# Physical constants & fixed seed for random functions
# -------------------------------------------------------------------------------

G_AU: float = const.G.to_value(u.AU**3 / (u.Msun * u.day**2)) # G in AU^3/(Msun*day^2)
AU_PER_DAY_KMS: float = (const.au / u.day).to(u.km / u.s).value # AU per day to km/s
RNG = np.random.default_rng(seed=42) # random number generator with a fixed seed for reproducibility
np.random.seed(42)  # Ensure reproducibility for np.random-based functions (which we used in mass functions)

G = const.G.to_value(u.m**3 / (u.kg * u.s**2)) # G in mks
c = const.c.to_value(u.m / u.s) # c in mks
pi = np.pi
# kg_to_SM = u.kg.to(u.Msun)


# ---------------------------------------------------------------------------
# Geometry helpers – isotropic + σ_theta‑banded directions
# ---------------------------------------------------------------------------

def isotropic_dirs(n: int) -> np.ndarray: # create isotropic unit vectors over the surface of a 3d sphere
    """
    phi = RNG.random(n) * 2.0 * np.pi
    cos_theta = RNG.random(n) * 2.0 - 1.0
    """
    phi = RNG.uniform(0.0, 2.0 * np.pi, n) # uniformly distributed azimuthal angle
    cos_theta = RNG.uniform(-1.0, 1.0, n) # uniformly distributed cosine of polar angle
    sin_theta = np.sqrt(1.0 - cos_theta**2) # sin(theta) from cos(theta)
    # np.column_stack returns unit vectors in (x, y, z) order
    return np.column_stack([ # restructure the separate variables into a single array (vector)
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

def banded_dirs(n: int, sigma_deg: float) -> np.ndarray:
    """
    Generate kick directions with Gaussian distribution about the orbital plane.
    
    Parameters:
    -----------
    n : int
        Number of direction vectors to generate
    sigma_deg : float
        1-sigma Gaussian latitude dispersion in degrees
        0° = perfectly in-plane, 90° = isotropic
    
    Returns:
    --------
    np.ndarray
        Array of shape (n, 3) containing unit direction vectors
    """
    # If sigma_deg is large (nearly 90°), return isotropic directions (no preference)
    if sigma_deg >= 89:
        return isotropic_dirs(n)
    
    # Generate theta (colatitude) values from a normal (Gaussian) distribution
    # Centered at 90° (the orbital plane), with standard deviation sigma_deg (converted to radians)
    theta = np.clip(
        RNG.normal(np.deg2rad(90), np.deg2rad(sigma_deg), n),  # sample n values
        0, np.pi  # restrict theta to [0, pi] (valid colatitude range)
    )
    # Generate azimuthal angle phi uniformly from 0 to 2*pi
    phi = RNG.uniform(0, 2 * np.pi, n)
    # Compute sin(theta) for conversion to Cartesian coordinates
    sin_theta = np.sin(theta)
    
    # Convert spherical coordinates (theta, phi) to Cartesian unit vectors (x, y, z)
    # x = sin(theta) * cos(phi)
    # y = sin(theta) * sin(phi)
    # z = cos(theta)
    return np.column_stack([
        sin_theta * np.cos(phi),  # x-component
        sin_theta * np.sin(phi),  # y-component
        np.cos(theta)             # z-component
    ])

# Maintain backward compatibility
def random_unit_vectors(n: int) -> np.ndarray:
    """Return an *(n, 3)* array of isotropically distributed unit vectors."""
    return isotropic_dirs(n)

# ------------------------------------------------------------------------------
#  Mass distribution functions
#-------------------------------------------------------------------------------

def m2_mass_function(
    batch_size,   # Batch size from mass_function
    m2_type,      # The type of the companion star to be created
    mu_NS = 1.33, # Mean mass of a neutron star
    std_NS = 0.05, # Standard deviation of neutron star mass
    mu_WD = 0.74,    # Mean mass of a white dwarf
    std_WD = 0.24,   # Standard deviation of white dwarf mass
    mu_MS = 3.6928642975,    # Mean mass of a main sequence star
    std_MS = 5.0467806977,   # Standard deviation of main sequence star mass
):
    """
    Generates a mass distribution for a companion star in a neutron star binary system.
    Required parameters:
    batch_size: int
        Number of companion stars to be generated.
    m2_type: str
        Type of the companion star to be generated. Can be 'NS', 'CO', 'He', or 'MS'.

    Optional parameters: see above (default values provided)
    """
    # Check m2_type validity
    if m2_type not in ['NS', 'MS', 'CO', 'He']:
        raise ValueError("m2_type must be one of 'NS', 'MS', 'CO', or 'He'")

    # Use a different distribution for each type
    if m2_type == 'NS': # Mass of a companion neutron star in a DNS system (Ozel et al. 2022)
        m2 = np.random.normal(loc=mu_NS, scale=std_NS, size=batch_size)
    elif m2_type == 'CO' or m2_type == 'He': # Mass of a companion white dwarf in a NS-WD system (Andrews et al. 2022)
        m2 = np.random.normal(loc=mu_WD, scale=std_WD, size=batch_size)
        # m2 = m2 > 1.44 # Chandrasekhar limit (I would probably return here, since adding clipping for a physical limit is kinda lazy)
    elif m2_type == 'MS': # Mass of a companion main sequence star in a NS-MS system (from diagrams provided)
        m2 = np.random.normal(loc=mu_MS, scale=std_MS, size=batch_size)

    return np.array(m2)

def mass_function(
    n, # Number of star systems to be generated
    m2_type, # The type of the companion star to be created
    alpha = -2.35, # Power-law index for the mass function (Salpeter function)
    zams_min = 8.0, # Initial mass of the primary star (before mass loss due to old age), minimum value
    zams_max = 25.0, # Initial mass of the primary star (before mass loss due to old age), maximum value
    a0 = -2.0, # Metallicity parameter 1 (for solar-like cases)
    a1 = 0.4, # Metallicity parameter 2 (for solar-like cases)
    a2 = 0.009, # Metallicity parameter 3 (for solar-like cases)
    iterations = 25, # Number of iterations before abandoning this function without a full list (increasing this parameter increases the chance of generating n systems at the cost of performance)
    m1_post_min = 1.0, # Minimum allowed mass for the primary star (for masking)
    m1_post_max = 2.25, # Maximum allowed mass for the primary star (for masking) [aka TOV limit]
    m2_min = 0.1, # Minimum allowed mass for the companion star (for masking)
    m2_max = 10.0 # Maximum allowed mass for the companion star (for masking)
):
    """
    Generates mass distributions for a neutron star binary system,
    where the primary star is a neutron star and the secondary star can be a neutron star, white dwarf, or main sequence star.
    Uses an evolutionary approach for modeling the mass of the primary star, and typical distributions for the companion star.
    Smart clipping is used to ensure n systems are generated.

    Required parameters:
    n: int
        Number of star systems to be generated.
    m2_type: str
        Type of the companion star to be generated. Can be 'NS', 'MS', 'CO', or 'He'.

    Optional parameters: see above (default values provided)
    """
    # Check m2_type validity
    if m2_type not in ['NS', 'MS', 'CO', 'He']:
        raise ValueError("m2_type must be one of 'NS', 'MS', 'CO', or 'He'")

    # Smart clipping parameters
    valid_systems = 0 # Number of valid systems collected so far
    iteration = 0 # Number of iterations performed
    m1_pre_valid = []
    m1_post_valid = []
    m2_valid = []

    # Main loop
    while valid_systems < n and iteration < iterations: # Smart clipping conditions. Stop either when n systems are generated, or when the threshold for the number of loop iterations is exceeded
        rejection_rate = 1 - (valid_systems / (iteration * batch_size)) if iteration > 0 else 0.5 # Rejection rate for dynamic batch size calculation, i.e., how many outputs are currently invalid per all outputs
        batch_size = int((n - valid_systems) / max(rejection_rate, 0.1)) # Batch size for current rate of rejections

        # Birth mass of the primary star (ZAMS mass) in a double neutron star system
        u = np.random.uniform(0, 1, batch_size) # Uniform random numbers for sampling the mass function
        zams = (u * (zams_max**(alpha + 1) - zams_min**(alpha + 1)) + zams_min**(alpha + 1))**(1 / (alpha + 1)) # ZAMS mass function of the primary star, i.e., birth mass (Malkov & Zinnecker 2001)

        # Pre-explosion mass of the primary star in a double neutron star system (when it is old)
        #mass_loss_factor = np.random.uniform(0.4, 0.55, batch_size) # Usual approximation for ZAMS mass loss (%) -- outdated --

        high_mass_loss = -0.035 * zams + 1.0667 # a linear function that goes from 0.55 at 15 to 0.2 at 25 through 0.35 at 20
        mass_loss_factor = np.where(zams < 15, 0.7, high_mass_loss)  # Approximation for mass-loss - ZAMS dependency (Meynet et al. 2015)
        m1_pre = zams * mass_loss_factor

        # Post-explosion mass of the primary star

        m_co = a0 + a1 * zams + a2 * zams**2 # Compact remnant mass (approximation using selected metallicity constants, Fryer et al. 2012)

        m_proto = np.select( # Proto-compact object mass (StarTrack simulation case, Fryer et al. 2012)
            [m_co < 4.82,
            m_co < 6.31,
            m_co < 6.75],
            [
                1.50,
                2.11,
                0.69 * m_co - 2.26
            ],
            default=(0.37 * m_co - 0.07)
        )

        f_fb = np.select( # Fallback factor for the mass loss (Fryer et al. 2012)
            [m_co < 5.00,
            m_co < 7.6],
            [
                0,
                0.378 * m_co - 1.889
            ],
            default=(1.0)
        )

        m_fb = np.where( # Fallback mass. Defined as a piecewise function (Fryer et al. 2012). Case for StarTrack simulations approximation and fit
            m_co < 5.0,
            0,
            f_fb * (m1_pre - m_proto)
        )

        m_rb = m_proto + m_fb # Final baryonic mass of the remnant after the supernova explosion (Fryer et al. 2012)

        # Gravitational mass from baryonic mass (StarTrack; Fryer et al. 2012, from Eq. 13 -- derived inverse equation)
        in_sqrt = (1 + 0.3 * m_rb) # Clip out negative roots
        m1_post = np.where(in_sqrt < 0, -1, ((-1 + np.sqrt(in_sqrt)) / 0.15))

        # For star 2, use the standard double-neutron star distribution to avoid overcomplicating the code
        m2 = m2_mass_function(batch_size, m2_type) # Mass of the companion star

        # Clipping conditions
        ns_mask = (m1_post >= m1_post_min) & (m1_post <= m1_post_max) & (m2 >= m2_min) & (m2 <= m2_max) # Mask to filter out irrelevant cases. Also removes invalid cases where m1 was marked as -1
        valid_indices = np.where(ns_mask)[0]

        # Actual clipping
        if len(valid_indices) > 0:
            # Take only what is needed (up to n total)
            take_count = min(len(valid_indices), n - valid_systems)
            m1_pre_valid.extend(m1_pre[valid_indices[:take_count]])
            m1_post_valid.extend(m1_post[valid_indices[:take_count]])
            m2_valid.extend(m2[valid_indices[:take_count]])
            valid_systems += take_count
        iteration += 1

    if valid_systems < n: # Warning for when not enough valid systems were generated; this could affect the main code
        print(f"Warning: Only generated {valid_systems} valid systems out of {n} requested after {iterations} iterations")

    return np.array(m1_pre_valid), np.array(m1_post_valid), np.array(m2_valid)

# ---------------------------------------------------------------------------
# Post-supernova survivability function for an elliptical case
# ---------------------------------------------------------------------------

# updated function will now have a case for elliptical orbits, deriving v_orb from Vis-Viva equation, and with f, e sampling added (as compared to the previous version).
# We set coordinate system in such a way that the explosion happens when r_vec = <r, 0, 0>, but r!=a_0, and velocity is not orthogonal to r_vec
# The specific angular momentum vector is computed via the cross product of position and velocity vectors, used to calculate the final eccentricity.

def post_sn_orbit(
    m1_pre: np.ndarray,          # mass of the primary star before SN {SM}
    m1_post: np.ndarray,         # mass of the primary star after SN {SM}
    m2: np.ndarray,              # mass of the companion star {SM}
    a0: np.ndarray,              # array of initial semi-major axes (not distances between bodies) {AU}
    v_kick_vec: np.ndarray,      # kick velocity vectors {km/s}
    f_ini: np.ndarray,           # true anomaly sampling (from low to high, uniform and random)
    e_ini: np.ndarray            # eccentricity sampling (from 0 to 1, uniform and random)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function simulates the post-supernova semi-major axis and eccentricity of a binary system. It now has circular and elliptical initial orbit cases.
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

# ---------------------------------------------------------------------------
# Monte-Carlo Wrapper – now requires sigma_theta param
# ---------------------------------------------------------------------------

def simulate_survival(
    n_samples: int = 10_000,
    *,
    f_low: float,
    sigma_low: float,
    sigma_high: float,
    sigma_theta: float,
    log_a0_mean: float,
    log_a0_sigma: float,
    m2_type, # the type of the companion star (change for other results, supports NS-MS and NS-WD, though we will need updated dns catallogue for such cases instead of the default one)
    plot_mass_distributions: bool = False # flag for plotting mass functions
) -> pd.DataFrame:
    """
    Generate *n_samples* supernova events and return outcomes in a DataFrame.
    
    Now uses the six-parameter model:
    1. f_low: Fraction of low-kick events
    2. sigma_low: Rayleigh σ of low-kick component (km/s)
    3. sigma_high: Rayleigh σ of high-kick component (km/s)
    4. sigma_theta: 1-σ Gaussian latitude dispersion about orbital plane (degrees)
    5. log_a0_mean: μ of log-normal pre-SN separation distribution
    6. log_a0_sigma: σ of that log-normal
    """
    # Generate initial orbital separations for each binary from a log-normal distribution
    a0 = RNG.lognormal(log_a0_mean, log_a0_sigma, n_samples)
    
    # Generate kick magnitudes for each event from a bimodal Rayleigh distribution
    # With probability f_low, use sigma_low; otherwise, use sigma_high
    vk_mag = np.where(
        RNG.random(n_samples) < f_low,  # Boolean mask: True for low-kick, False for high-kick
        rayleigh.rvs(scale=sigma_low, size=n_samples, random_state=RNG),  # Low-kick component
        rayleigh.rvs(scale=sigma_high, size=n_samples, random_state=RNG)  # High-kick component
    )
    
    # Generate kick direction unit vectors using a banded (possibly anisotropic) distribution
    k_dir = banded_dirs(n_samples, sigma_theta)
    # Multiply each direction by its magnitude to get the full 3D kick velocity vectors
    vk_vec = vk_mag[:, None] * k_dir
    
    # Calculate spherical coordinates (theta, phi) of each kick for diagnostics
    epsilon = 1e-9  # Small value to avoid division by zero
    theta = np.arccos(np.clip(vk_vec[:, 2] / (vk_mag + epsilon), -1, 1))  # Polar angle from z-axis
    phi = np.arctan2(vk_vec[:, 1], vk_vec[:, 0])  # Azimuthal angle in x-y plane

    # mass function call
    # NOTE: to make NS-WD or NS-MS instead of DNS, change m2_type value in the function call
    m1_pre, m1_post, m2 = mass_function(n_samples, m2_type)

    # mass distribution plotting
    if plot_mass_distributions and plt is not None: # plots when not in ABC (so we dont generate appr. 1000 plots for each ABC trial)
        num_bins = 50  # Number of bins for histograms
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(m1_pre, bins=num_bins, color='skyblue', edgecolor='k')
        plt.title('Pre-explosion mass (m1_pre)')
        plt.xlabel('Mass [M_sun]')
        plt.ylabel('Count')

        plt.subplot(1, 3, 2)
        plt.hist(m1_post, bins=num_bins, color='salmon', edgecolor='k')
        plt.title('Post-explosion mass (m1_post)')
        plt.xlabel('Mass [M_sun]')

        plt.subplot(1, 3, 3)
        plt.hist(m2, bins=num_bins, color='lightgreen', edgecolor='k')
        plt.title(f'Companion mass (m2)\nType: {m2_type}')
        plt.xlabel('Mass [M_sun]')

        plt.tight_layout()
        plt.show()

    # true anomaly and eccentricity sampling

    f_ini = np.random.uniform(0, 2 * np.pi, n_samples) # True anomaly sampling (from 0 to 2π, uniform*)
    # * -- assumed to be equally possible to explode at any point in the orbit, i.e., no preference for the explosion point coordinate
    mu_e = 0.0 # pre-DNS binaries have a very low eccentricity, since tidal effects and mass transfer often circularize the orbit
    sigma_e = 0.1 # standard deviation for eccentricity sampling (assumed to be small, i.e., 0.1)
    e_ini = np.random.normal(loc=mu_e, scale=sigma_e)

    # Run the core orbital physics simulation to determine if the system remains bound
    # and to compute the final orbital parameters after the supernova kick
    bound, a_fin, e_fin = post_sn_orbit(m1_pre, m1_post, m2, a0, vk_vec, f_ini, e_ini)

    # Return all relevant results as a pandas DataFrame
    # Each row corresponds to one simulated binary system
    return pd.DataFrame({
        "a0": a0,           # Initial separation (AU)
        "v_kick": vk_mag,   # Kick magnitude (km/s)
        "vk_x": vk_vec[:, 0],  # Kick x-component (km/s)
        "vk_y": vk_vec[:, 1],  # Kick y-component (km/s)
        "vk_z": vk_vec[:, 2],  # Kick z-component (km/s)
        "theta": theta,     # Kick polar angle (radians)
        "phi": phi,         # Kick azimuthal angle (radians)
        "bound": bound,     # Boolean: True if system remains bound
        "a_fin": a_fin,     # Final semi-major axis (AU), NaN if unbound
        "e_fin": e_fin,     # Final eccentricity, NaN if unbound
    })

# ---------------------------------------------------------------------------
# ATNF DNS Catalog, correction for gravitational circularization (a, e values)
# ---------------------------------------------------------------------------

def a0_retrieval(
        name,  # the name of the pulsar
        ecc_f,  # final eccentricity
        m2,     # the median mass (i=60 deg) of the companion star {sm}
        m2_min, # the minimum mass (i=90 deg) of the companion star {sm}
        pb,   # orbital period {days}
        age_i, # the age since SN {years}
        print_dt # flag to print (or not) the data table
):
    """
    This function applies the a0_retrieval_single function to a list of pulsars. For a large list, the function is unuseable as it uses 'for' loop.
    The datatable of all circularization parameters is printed as well. It is set to print only median (not minimum) values by defualt.
    """
    ecc_ini_all = np.zeros(len(name)) # initial eccentricity {dimensionless}
    a_ini_all = np.zeros(len(name)) # initial semimajor axis {m}

    if print_dt:
        print('\n\n\nData table for e, a corrections:') # for data table printing only
        print('Note: final means from observed data, initial means before circularization\n')
        print('name | (ecc_ini, ecc_fin) | (a_ini, a_fin)')

    # zip needed to run over a frame of data instead of a single value like ecc_f
    for i, (name_curr, ecc_f_curr, m2_curr, pb_curr, age_i_curr) in enumerate(zip(name, ecc_f, m2, pb, age_i)):
        e_ini, a_ini = a0_retrieval_single(name_curr, ecc_f_curr, m2_curr, pb_curr, age_i_curr, print_dt)
        ecc_ini_all[i] = e_ini
        a_ini_all[i] = a_ini

    # second loop for minimum mass values
    ecc_ini_all_min = np.zeros(len(name))
    a_ini_all_min = np.zeros(len(name))
    for i, (name_curr, ecc_f_curr, m2_min_curr, pb_curr, age_i_curr) in enumerate(zip(name, ecc_f, m2_min, pb, age_i)):
        e_ini, a_ini = a0_retrieval_single(name_curr, ecc_f_curr, m2_min_curr, pb_curr, age_i_curr, print_dt2=False)
        ecc_ini_all_min[i] = e_ini
        a_ini_all_min[i] = a_ini

    if print_dt:
        print('\n\n')
    return ecc_ini_all, a_ini_all, ecc_ini_all_min, a_ini_all_min

def a0_retrieval_single(
        name,  # the name of the pulsar
        ecc_f,  # final eccentricity
        m2,     # the mass of the companion star {sm}
        pb,   # orbital period {days}
        age_i, # the age since SN {years}
        print_dt2 # flag for when to print the data
):
    """
    this function corrects the eccentricity for the circularization effects.
    Using da_dt and de_dt, the function goes back in time (dt < 0) from a1_final and e_final to a1_initial and e_initial. For the correctness of numerical integration with a negative dt, 
    Runge-Kutta 4th order method of numerical integration is used.

    parameters:
        required:
            ecc_f: the final eccentricity
            m2: the mass of the companion star {sm}
            pb: the orbital period {days}
            age_i: the age since SN {years}
        optional:
            name: the name of the psrj
            m1: the mass of the post_SN neutron star {sm}. assumed to be 1.35 SM by default

    returns:
        e_ini: the initial eccentricity 
        a_ini: the initial semimajor axis {light seconds}

    """

    # unit conversions and pre-integration calculations

    m1 = (1.35 * u.Msun).to_value(u.kg) # the main star in all NS systems (DNS, NS-MS, NS-CO, NS-He) is just a neutron star with a mass close to 1.35 SM. trnasformed to kg
    m2 = (m2 * u.Msun).to_value(u.kg) # the companion star mass {kg}
    pb = (pb * u.day).to_value(u.s) # {sec}
    age_i_sec = (age_i * u.year).to_value(u.s) # {sec}

    M = m1 + m2 # total mass {kg}
    a1_f = ((G * M * pb**2) / (4 * np.pi**2)) ** (1.0/3.0) # final semimajor axis {m}
    # NOTE: we still use e, m1, m2 -- all of which are dependent on sin(i), so calculating a1 through Kepler's 3rd law only slightly decreases the degree of uncertainty

    # functions to integrate (since they depend on each other, for dual ODEs, they are combined)

    def gw_derivs(t, y, m1, m2):
        a, ecc = y # state vector for dual ODEs
        da = -(64/5) * (G**3 / c**5) * (m1 * m2 * (m1 + m2) / a**3) * ((1 + (73/24) * ecc**2 + (37/96) * ecc**4) / (1 - ecc**2)**3.5)
        de = (-304/15) * ((G**3)/(c**5)) * ((m1 * m2 * (m1 + m2))/(a**4)) * ((ecc * (1 + (121/304) * ecc**2))/((1 - ecc**2)**2.5))
        return [da, de]

    def gw(t, y): return gw_derivs(t, y, m1, m2)

    #da_dt = lambda t, a, ecc: -(64/5) * (G**3 / c**5) * (m1 * m2 * (m1 + m2) / a**3) * ((1 + (73/24) * ecc**2 + (37/96) * ecc**4) / (1 - ecc**2)**3.5)
    #de_dt = lambda t, ecc, a: (-304/15) * ((G**3)/(c**5)) * ((m1 * m2 * (m1 + m2))/(a**4)) * ((ecc * (1 + (121/304) * ecc**2))/((1 - ecc**2)**2.5))

    # THE integration through scipy tools

    sol = solve_ivp(
        gw_derivs,
        t_span=[0, -1 * age_i_sec], # integrate back in time for the initial values from the final ones
        y0=[a1_f, ecc_f],
        args=(m1, m2), # extra parameters for gw_derivs (constant for a single iteration of a0_single)
        rtol=1e-8, atol=1e-10, # accuracy
        dense_output=False)
    a_ini, e_ini = sol.y[:, -1]
    a_ini = a_ini / c # convert to light seconds
    a1_f = a1_f / c # cpnvert to light seconds (as in the catalogue)

    # only for printing the table of data differences
    datatable = [name, (e_ini, ecc_f), (a_ini, a1_f)]
    print(datatable) if print_dt2 else None

    return e_ini, a_ini


def load_dns_catalogue(
        plot_type, # whether the plot will be with a log y-axis or standard y-axis
        m2_type, # the type of the companion star for the correct catalogue
        path, # the pathe to the catalogue file
        print_dt=False,
        catalogue_types = ('NS', 'MS', 'CO', 'He'),
        cache_file: str = "dns_*_catalogue.csv"
):
    """
    Loads observed Neutron Star binary data (for NS-MS, DNS, NS-CO, NS-He), using the separate function for correcting eccentricities.

    parameters:
        required:
            plot_type: ['log', 'standard'] - specifies the y-axis that the plot will have
            m2_type: the type of the companion star for the correct catalogue
            cache_file: the path to the cache file
        optional:
            catalogue_types: the types of the companion stars for the correct catalogue. ('NS', 'MS', 'CO', 'He') by default
            print_dt: boolean, indicates whether the data table of corrected circularization effects would be printed. False by default

    returns:
        name: the name of the pulsar
        ecc: the eccentricity of the pulsar (corrected for initial values)

    plots:
        a graph of current ecc values against initial ones with the margin of error from uncertain inclination angle of DNS systems.

    """
    # This function loads observed Double Neutron Star (DNS) data.
    # then the data is corrected for circularization effects
    # the result is now returned as a simple list

    if path and path[-1] != '/': path += '/'
    cache_file = path + cache_file if path else cache_file
    #print(f'\nyour total path is {cache_file}\n')
    if m2_type == 'NS' and "NS" in catalogue_types:
        cache_file = cache_file.replace('*', 'NS')
    elif m2_type == 'MS' and "MS" in catalogue_types:
        cache_file = cache_file.replace('*', 'MS')
    elif m2_type == 'CO' and "CO" in catalogue_types:
        cache_file = cache_file.replace('*', 'CO')
    elif m2_type == 'He' and "He" in catalogue_types:
        cache_file = cache_file.replace('*', 'He')
    else:
        raise ValueError(f'{m2_type} not in catalogue types. Catalogue types are {catalogue_types}.')
    # He and CO are subtypes for NS-WD systems

    print(f"Loading observed DNS data from local cache: {cache_file}")
    try:
        df = pd.read_csv(cache_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {cache_file}. Double-check the path correctness")

    name = df['PSRJ']
    ecc = df['ECC'] # dimentionless
    medmass = df['MEDMASS'] # in solar masses. The mass of the companion star when i=60 deg
    minmass = df['MINMASS'] # in solar masses. The mass of the companion star when i=90 deg
    pb = df['PB'] # in days
    age_i = df['AGE_I'] # in years

    mask = (ecc > 0) & (medmass > 0) & (pb > 0) & (age_i > 0) & (minmass > 0) # Select only double neutron star systems (BINCOMP starts with 'NS') and non-null eccentricity
    name_m = name[mask]
    ecc_m = ecc[mask]
    medmass_m = medmass[mask]
    pb_m = pb[mask]
    age_i_m = age_i[mask]
    minmass_m = minmass[mask]

    ecc_m_corr, a_m_corr, ecc_m_corr_min, a_m_corr_min = a0_retrieval(name_m, ecc_m, medmass_m, minmass_m, pb_m, age_i_m, print_dt) # correct the eccentricity to the intiial post explosion ones

    plt.figure(figsize=(10, 6))
    x = np.arange(len(ecc_m_corr))
    y = ecc_m_corr
    
    # imaginary line to divide the data set into two areas
    x_ = np.linspace(0.0, (float(len(ecc_m_corr)) + 1.00), 1000)
    y_ = 0*x_ + 0.0167 # symbolizes Earth's eccentricity. x_ added to make y_ the same size as x_
    plt.plot(x_, y_, color='black', label='Earth eccentricity')
    
    if plot_type == 'log' and plt:
        plt.semilogy(np.arange(len(ecc_m)), ecc_m, marker='x', linestyle='none', color='green', label='Current observed eccentricity')
        plt.semilogy(x, y, marker='.', linestyle='none', color='blue', label='Eccentricity after circularization', alpha=0.7)
        plt.semilogy(x, ecc_m_corr_min, marker='+', linestyle='none', color='blue', label='Eccentricity low bound', alpha=0.6)
        for i in range(len(x)):
            plt.plot([x[i], x[i]], [ecc_m_corr_min[i], y[i]], color='gray', alpha=0.5, linewidth=5)
        plt.xlabel('DNS System #')
        plt.ylabel('Eccentricity')
        plt.title(f'Distribution of Initial and Final Eccentricities for {m2_type}-DNS Systems')
        plt.legend()
        plt.show()
    elif plot_type == 'standard' and plt:
        plt.scatter(np.arange(len(ecc_m)), ecc_m, marker='x', color='green', label='Current observed eccentricity')
        plt.scatter(x, y, marker='.', color='blue', label='Eccentricity after circularization', alpha=0.7)
        plt.scatter(x, ecc_m_corr_min, marker='+', color='blue', label='Eccentricity low bound', alpha = 0.6)
        for i in range(len(x)):
            plt.plot([x[i], x[i]], [ecc_m_corr_min[i], y[i]], color='gray', alpha=0.5, linewidth=5)
        plt.xlabel('DNS System #')
        plt.ylabel('Eccentricity')
        plt.title(f'Distribution of Initial and Final Eccentricities for {m2_type}-DNS Systems')
        plt.legend()
        plt.show()
    else:
        if plt: print(f'Warning: plot type should be "log" or "standard". Yours is {plot_type}')

    if len(name_m) == 0 or len(ecc_m_corr) == 0:
        print("Warning: No NS binary systems were found")
    df_out = pd.DataFrame({
        'PSRJ': name_m.values,
        'ECC': ecc_m_corr
    })
    return df_out

# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def plot_simulation_results(m2_type, sim_df: pd.DataFrame, obs_df: pd.DataFrame, title_suffix=""):
    """Generates plots to compare simulated and observed populations."""
    # Check if plotting libraries are available
    if plt is None: return
    # Select only surviving (bound) systems from the simulation
    survivors = sim_df[sim_df.bound].dropna()
    if survivors.empty:
        print("\nNo systems survived. Cannot generate plots."); return
    if obs_df.empty:
        print("\nObserved data is empty. Cannot generate plots."); return

    print(f"\nGenerating simulation outcome plots for {len(survivors)} surviving systems...")
    sns.set_theme(style="whitegrid", context="talk")
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    base_title = "Comparison of Simulated vs. Observed Double Neutron Stars"
    fig.suptitle(f"{base_title}{title_suffix}", fontsize=20)

    # --- Plot 1: Histogram of eccentricity distributions ---
    ax = axes[0]
    DNSt = f"NS-{m2_type}" if m2_type != "NS" else "DNS"
    sns.histplot(
        obs_df.ECC, ax=ax, stat="density", bins=np.linspace(0, 1, 15),
        color="C0", alpha=0.6, label=f"Observed {DNSt} ({len(obs_df)})"
    )
    sns.histplot(
        survivors.e_fin, ax=ax, stat="density", bins=np.linspace(0, 1, 15),
        color="C1", alpha=0.6, label=f"Simulated Survivors ({len(survivors)})"
    )
    ax.set_xlabel("Orbital Eccentricity (e)")
    ax.set_title("Eccentricity Distribution")
    ax.legend()
    ax.set_xlim(0, 1)

    # --- Plot 2: Scatter plot of final semi-major axis vs eccentricity, colored by kick velocity ---
    ax = axes[1]
    sns.scatterplot(
        x=survivors.a_fin, y=survivors.e_fin, hue=survivors.v_kick,
        palette="viridis", ax=ax, alpha=0.7, s=30,
    )
    ax.set_xlabel("Final Semi-Major Axis (AU)")
    ax.set_ylabel("Final Eccentricity (e)")
    ax.set_title("Properties of Surviving Systems")
    ax.set_xscale("log")
    ax.legend(title="Kick Vel (km/s)")
    plt.show()

# ---------------------------------------------------------------------------
# Main Visualization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Kick Survival Monte Carlo Demo ---")

    try:
        # Step 1: Load the observed Double Neutron Star (DNS) catalogue.
        # This will use a local file. File installation required
        print("\nStep 1: Loading observed Double Neutron Star catalogue...")


        # !!! NOTE: modifiable parameters below. 
        m2_type = 'NS' # change m2_type here. Possible values presented in the catalogue types tuple below.
        catalogue_types = ('NS', 'MS', 'CO', 'He')
        path = "" # set the path here as a string if not in the same firectory. Do not include the filename. Example: 'C:/user/folder/' 
        plot_type = 'log' # plot type for the circularization plots. Can be ['log', 'standard'] only.
        print_dt = False # boolean, indicates whether to print the data table of corrected circularization effects.
        print(f'm2 type selected is {m2_type}. \nPlot type is {plot_type}. \nYour path to the catalog is {path}\n\n')

        if m2_type not in catalogue_types:
            raise ValueError(f"m2_type must be one of {catalogue_types}.")
        if path is None:
            print('Warning: The path to catalog file is not specified.')
        if plot_type is None:
            raise ValueError('No plot type provided.')
        if print_dt not in (True, False):
            raise ValueError('print_dt must be True or False.')
        
        dns_df_corr = load_dns_catalogue(plot_type, m2_type, path=path, catalogue_types=catalogue_types, print_dt=print_dt)
        if dns_df_corr is None or dns_df_corr.empty:
            raise ValueError(f"No {m2_type} DNS systems found in the catalogue. Check the data source file.")
        else:
            print(f"Successfully loaded {len(dns_df_corr)} confirmed DNS with eccentricity data.")

            # Step 2: Run a forward simulation with default parameters.
            # This generates a synthetic population of binaries using the model.
            print("\nStep 2: Running forward simulation with 50,000 samples...")
            df_sim = simulate_survival(
                200_000,                # Number of Monte Carlo samples (binaries)
                f_low=0.7,             # Fraction of low-kick events (bimodal kick model)
                sigma_low=30.0,        # Rayleigh sigma for low-kick component (km/s)
                sigma_high=300.0,      # Rayleigh sigma for high-kick component (km/s)
                sigma_theta=45.0,      # 1-sigma Gaussian latitude dispersion (degrees)
                log_a0_mean=1.44, # Mean of log-normal initial separation (AU, in log-space)
                log_a0_sigma=1.53,      # Sigma of log-normal initial separation (log-space)
                plot_mass_distributions=True, # plots only when not in ABC
                m2_type=m2_type
            )
            bound_frac = df_sim.bound.mean()
            print(f"-> Simulation complete. Bound fraction ≈ {bound_frac:.3f}")

            # Step 3: Plot diagnostic plots of the initial conditions.
            # This visualizes the initial separation and kick velocity distributions, including 3D kicks.
            # NOTE: uncomment the next line to enable initial condition plots.

            #print("\nStep 3: Generating diagnostic plots for initial conditions...")
            #plot_initial_conditions(df_sim)

            # Step 4a: Plot the comparison between simulated and observed populations.
            # This overlays the simulated and observed eccentricity distributions and other properties.
            plot_simulation_results(m2_type, df_sim, dns_df_corr, title_suffix=" (Default Model)")

            # Step 4b: Simulate with no kicks and plot the results as well.
            print("\nStep 4b: Running simulation with NO kicks...")
            df_nokick = simulate_survival(
                200_000,
                f_low=1.0,            # All events are low-kick (but sigma=0)
                sigma_low=0.0,        # No kick
                sigma_high=0.0,       # No kick
                sigma_theta=0.0,      # No directional dispersion
                log_a0_mean=np.log(0.05),
                log_a0_sigma=0.5,
                plot_mass_distributions=False,
                m2_type=m2_type
            )

            # no-kick case diagnostics (since no survivors per this seed)
            print(f"-> No-kick simulation complete. Bound fraction ≈ {df_nokick.bound.mean():.3f}")
            # Diagnostics for no-kick case
            survivors = df_nokick[df_nokick.bound]
            print(f"No-kick survivors: {len(survivors)} out of {len(df_nokick)}")
            print(f"Mean initial separation (AU): {df_nokick['a0'].mean():.3f}")
            # To get masses,  mass_function is run for the same n (fixed seed, so the result would be the same)
            m1_pre, m1_post, m2 = mass_function(200000, m2_type)
            print(f"Mean pre-SN mass of the main star: {m1_pre.mean():.3f}, Mean post-SN mass of the main star: {m1_post.mean():.3f}, Mean companion mass: {m2.mean():.3f}")
            print(f"Mean total pre-SN mass: {(m1_pre + m2).mean():.3f}, Mean total post-SN mass: {(m1_post + m2).mean():.3f}")
            print(f"Fractional mass loss (mean): {((m1_pre + m2).mean() - (m1_post + m2).mean()) / (m1_pre + m2).mean():.3f}")
            print("For losses above 0.5, the system cannot remain bound, which seems to be the case here.\nNote that increasing mass loss factor will create a few surviving systems")
            plot_simulation_results(m2_type, df_nokick, dns_df_corr, title_suffix=" (No Kicks)")

    except Exception as e:
        # If any error occurs during the demo, print the error and traceback for debugging.
        import traceback
        print(f"\n\nAn error occurred during the demonstration part of the code: {e}")
        traceback.print_exc()
        print("\nPlease check catalogue installation and path, and that all required libraries are installed.")

    print("\n--- Visualization Finished ---")