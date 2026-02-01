import numpy as np

# Mass functions

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
        Type of the companion star to be generated. Can be 'DNS', 'NS-WD', or 'NS-MS'.

    Optional parameters: see above (default values provided)
    """
    # Check m2_type validity
    if m2_type not in ['DNS', 'NS-WD', 'NS-MS']:
        raise ValueError("m2_type must be one of 'DNS', 'NS-WD', or 'NS-MS'")

    # Use a different distribution for each type
    if m2_type == 'DNS': # Mass of a companion neutron star in a DNS system (Ozel et al. 2022)
        m2 = np.random.normal(loc=mu_NS, scale=std_NS, size=batch_size)
    elif m2_type == 'NS-WD': # Mass of a companion white dwarf in a NS-WD system (Andrews et al. 2022)
        m2 = np.random.normal(loc=mu_WD, scale=std_WD, size=batch_size)
    elif m2_type == 'NS-MS': # Mass of a companion main sequence star in a NS-MS system (from diagrams provided)
        m2 = np.random.normal(loc=mu_MS, scale=std_MS, size=batch_size)

    return np.array(m2)

def mass_function(
    n, # Number of star systems to be generated
    m2_type, # The type of the companion star to be created
    alpha = -2.35, # Power-law index for the mass function (Salpeter function)
    zams_min = 8.0, # Initial mass of the primary star (before mass loss due to old age), minimum value
    zams_max = 20.0, # Initial mass of the primary star (before mass loss due to old age), maximum value
    a0 = -2.0, # Metallicity parameter (for solar-like cases)
    a1 = 0.4, # Metallicity parameter (for solar-like cases)
    a2 = 0.009, # Metallicity parameter (for solar-like cases)
    iterations = 25, # Number of iterations before abandoning this function without a full list (increasing this parameter increases the chance of generating n systems at the cost of performance)
    m1_post_min = 1.0, # Minimum allowed mass for the primary star (for masking)
    m1_post_max = 2.25, # Maximum allowed mass for the primary star (for masking)
    m2_min = 0.1, # Minimum allowed mass for the companion star (for masking)
    m2_max = 8.0 # Maximum allowed mass for the companion star (for masking)
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
        Type of the companion star to be generated. Can be 'DNS', 'NS-WD', or 'NS-MS'.

    Optional parameters: see above (default values provided)
    """
    # Check m2_type validity
    if m2_type not in ['DNS', 'NS-WD', 'NS-MS']:
        raise ValueError("m2_type must be one of 'DNS', 'NS-WD', or 'NS-MS'")

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

        mass_loss_factor = np.where(zams < 15, 0.7, np.random.uniform(0.4, 0.6, batch_size))  # Approximation for mass-loss - ZAMS dependency (Meynet et al. 2015)
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

        m_fb = np.where( # Fallback mass. Defined as a piecewise function (Fryer et al. 2012). Case for instant supernovae
            m_co < 5.0,
            0,
            f_fb * (m1_pre - m_proto)
        )

        m_rb = m_proto + m_fb # Final baryonic mass of the remnant after the supernova explosion (Fryer et al. 2012)

        # Gravitational mass from baryonic mass (StarTrack; Fryer et al. 2012, Eq. 13)
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
        raise RuntimeWarning(f"Warning: Only generated {valid_systems} valid systems out of {n} requested after {iterations} iterations")

    return np.array(m1_pre_valid), np.array(m1_post_valid), np.array(m2_valid)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    import matplotlib.pyplot as plt

    m2_type = 'DNS' # Type of the companion to be generated. Can be 'DNS', 'NS-WD', or 'NS-MS'
    n = 50000
    m1_pre, m1_post, m2 = mass_function(n, m2_type)

    # Print summary statistics
    print("Pre-explosion mass of the main star (m1_pre):", np.round(m1_pre, 3))
    print("Post-explosion mass of the main star (m1_post):", np.round(m1_post, 3))
    print(f"Mass of the companion star in {m2_type} (m2): {np.round(m2, 3)}")

    # Plot distributions
    num_bins = max(10, n // 1000)
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