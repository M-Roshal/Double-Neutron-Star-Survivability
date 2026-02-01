# ecc correction function

# https://www.google.com/url?q=https%3A%2F%2Fjournals.aps.org%2Fpr%2Fpdf%2F10.1103%2FPhysRev.136.B1224
# Peter 1964 - the paper used for calculating circularization effects from gravitational radiation

import pandas as pd
import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import solve_ivp

G = const.G.to_value(u.m**3 / (u.kg * u.s**2)) # G in mks
c = const.c.to_value(u.m / u.s) # c in mks
pi = np.pi
# kg_to_SM = u.kg.to(u.Msun)

def a0_retrieval(
        name,  # the name of the pulsar
        ecc_f,  # final eccentricity
        m2,     # the mass of the companion star {sm}
        pb,   # orbital period {days}
        age_i, # the age since SN {years}
):
    """
    This function applies the a0_retrieval_single function to a list of pulsars. For a large list, the function is unuseable as it uses 'for' loop.
    """
    ecc_ini_all = np.zeros(len(name)) # initial eccentricity {dimensionless}
    a_ini_all = np.zeros(len(name)) # initial semimajor axis {m}

    # zip needed to run over a frame of data instead of a single value like ecc_f
    for i, (name_curr, ecc_f_curr, m2_curr, pb_curr, age_i_curr) in enumerate(zip(name, ecc_f, m2, pb, age_i)):
        e_ini, a_ini = a0_retrieval_single(name_curr, ecc_f_curr, m2_curr, pb_curr, age_i_curr)
        ecc_ini_all[i] = e_ini
        a_ini_all[i] = a_ini

    return ecc_ini_all, a_ini_all

def a0_retrieval_single(
        name,  # the name of the pulsar
        ecc_f,  # final eccentricity
        m2,     # the mass of the companion star {sm}
        pb,   # orbital period {days}
        age_i, # the age since SN {years}
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
        args=(m1, m2), # extra parameters for gw_derivs
        rtol=1e-8, atol=1e-10, # accuracy
        dense_output=False)
    a_ini, e_ini = sol.y[:, -1]
    a_ini = a_ini / c # convert to light seconds

    return e_ini, a_ini


def load_dns_catalogue(
        m2_type, # the type of the companion star for the correct catalogue
        catalogue_types = ['NS', 'MS', 'CO', 'He'],
        cache_file: str = "dns_*_catalogue.csv"
):
    """
    Loads observed Neutron Star binary data (for NS-MS, DNS, NS-CO, NS-He), using the separate function for correcting eccentricities.

    parameters:
        required:
            m2_type: the type of the companion star for the correct catalogue
            cache_file: the path to the cache file
        optional:
            catalogue_types: the types of the companion stars for the correct catalogue

    returns:
        name: the name of the pulsar
        ecc: the eccentricity of the pulsar (corrected for initial values)

    """
    # This function loads observed Double Neutron Star (DNS) data.
    # then the data is corrected for circularization effects
    # the result is now returned as a simple list

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
        print(f"Error: File {cache_file} not found.")
        return None

    name = df['PSRJ']
    ecc = df['ECC'] # dimentionless
    medmass = df['MEDMASS'] # in solar masses
    pb = df['PB'] # in days
    age_i = df['AGE_I'] # in years

    mask = (ecc > 0) & (medmass > 0) & (pb > 0) & (age_i > 0) # Select only double neutron star systems (BINCOMP starts with 'NS') and non-null eccentricity
    name_m = name[mask]
    ecc_m = ecc[mask]
    medmass_m = medmass[mask]
    pb_m = pb[mask]
    age_i_m = age_i[mask]

    ecc_m_corr, a_m_corr = a0_retrieval(name_m, ecc_m, medmass_m, pb_m, age_i_m) # correct the eccentricity to the intiial post explosion ones

    # If no DNS systems found, warn and return empty DataFrame
    if len(name_m) == 0 or len(ecc_m_corr) == 0:
        print("Warning: No Double Neutron Star systems were found in the query.")

    exit_data = [name_m, ecc_m_corr] # return corrected eccentricity and names
    return exit_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    filename = "dns_*_catalogue.csv"
    m2_type = 'NS'
    filename = filename.replace('*', m2_type)
    dff = pd.read_csv(filename)
    name, ecc, medmass, pb, age_i = dff['PSRJ'], dff['ECC'], dff['MEDMASS'], dff['PB'], dff['AGE_I']
    mask = (ecc > 0) & (medmass > 0) & (pb > 0) & (age_i > 0)
    final_ecc = ecc[mask]

    name_m, initial_ecc = load_dns_catalogue(m2_type)
 
    plt.figure(figsize=(10, 6))
    plt.hist(initial_ecc, bins=len(initial_ecc), alpha=0.5, label='Initial Eccentricity', color='blue')
    plt.hist(final_ecc, bins=len(final_ecc), alpha=0.5, label='Final Eccentricity', color='red')
    plt.xlabel('Eccentricity')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Initial and Final Eccentricities for {m2_type}-DNS Systems')
    plt.legend()
    plt.show()