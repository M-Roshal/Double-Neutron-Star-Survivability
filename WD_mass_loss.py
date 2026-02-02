"""This program is going to be a variation of Kick_survival for WD uneven mass loss and graphs.

Contents: 

1. Enhanced e-P diagram for NS-WD(He) systems with circularization reversal and inclination uncertainty removal
2. "Line" power-law fit for gc systems in e-P diagram
3. WD mass function theory
4. One specific example simulation

Instructions:

"""
print('\n')

# ---- libraries and dependecies ----

import numpy as np
import pandas as pd

from astropy import constants as const
from astropy import units as u
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


# ---- global constants ----

G_AU: float = const.G.to_value(u.AU**3 / (u.Msun * u.day**2)) # G in AU^3/(M_sun*day^2)
AU_PER_DAY_KMS: float = (const.au / u.day).to(u.km / u.s).value
RNG = np.random.default_rng(seed=42) # fixed random seed
np.random.seed(42)

G = const.G.to_value(u.m**3 / (u.kg * u.s**2)) # G in mks
c = const.c.to_value(u.m / u.s) # c in mks
pi = np.pi
kg_to_SM = u.kg.to(u.Msun)


# ---- circularization reversal & inclination uncertainty removal ----

def a0_retrieval(
        name,  # the name of the pulsar
        ecc_f,  # final eccentricity
        m2,     # the median mass (i=60 deg) of the companion star {sm}
        m2_min, # the minimum mass (i=90 deg) of the companion star {sm}
        m2_max, # the maximum mass of 1.44 SM for WDs
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

    # third loop for maximum mass values
    ecc_ini_all_max = np.zeros(len(name))
    a_ini_all_max = np.zeros(len(name))
    for i, (name_curr, ecc_f_curr, m2_max_curr, pb_curr, age_i_curr) in enumerate(zip(name, ecc_f, m2_max, pb, age_i)):
        e_ini, a_ini = a0_retrieval_single(name_curr, ecc_f_curr, m2_max_curr, pb_curr, age_i_curr, print_dt2=False)
        ecc_ini_all_max[i] = e_ini
        a_ini_all_max[i] = a_ini

    if print_dt:
        print('\n\n')

    return ecc_ini_all, a_ini_all, ecc_ini_all_min, a_ini_all_min, ecc_ini_all_max, a_ini_all_max

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


# ---- data upload (catalogues) ----

def load_dns_catalogue(
        plot_type: str = "log", # whether the plot will be with a log y-axis or standard y-axis
        path: str = "", # the pathe] to the catalogue file
        print_dt=False,
        cache_file: str = "dns_He_catalogue.csv",
        grid: bool = True,
        font: str = 'medium'
):
    """
    Loads observed Neutron Star binary data (for NS-MS, DNS, NS-CO, NS-He), using the separate function for correcting eccentricities.

    parameters:
        required:
            plot_type: ['log', 'standard'] - specifies the y-axis that the plot will have
            cache_file: the path to the cache file (here He set and catalogues_types removed)
        optional:
            print_dt: boolean, indicates whether the data table of corrected circularization effects would be printed. False by default

    returns:
        name: the name of the pulsar
        ecc: the eccentricity of the pulsar (corrected for initial values)

    plots:
        A consolidated Eccentricity-Period (e-P) diagram for NS-WD(He) systems,
        comparing Galactic Field and Globular Cluster populations.
    """

    if path and path[-1] != '/': path += '/'
    cache_file = path + cache_file if path else cache_file

    print(f"Loading observed DNS data from local cache: {cache_file}")
    try:
        df = pd.read_csv(cache_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {cache_file}. Double-check the path correctness")
    
    gf_systems = [ # Ns_WD(He) in galactic field -- more stable in e-P
        'J0034-0534', 'J0101-6422', 'J0203-0150', 'J0214+5222', 'J0218+4232',
        'J0337+1715', 'J0348+0432', 'J0407+1607', 'J0437-4715', 'J0509+0856',
        'J0557+1550', 'J0605+3757', 'J0613-0200', 'J0614-3329', 'J0621+2514',
        'J0732+2314', 'J0740+6620', 'J0742+4110', 'J0751+1807', 'J0921-5202',
        'J0925+6103', 'J1012+5307', 'J1012-4235', 'J1017-7156', 'J1045-4509',
        'J1056-7117', 'J1120-3618', 'J1125+7819', 'J1125-5825', 'J1125-6014',
        'J1137+7528', 'J1142+0119', 'J1146-6610', 'J1216-6410', 'J1231-1411',
        'J1232-6501', 'J1312+0051', 'J1327-0755', 'J1400-1431', 'J1405-4656',
        'J1421-4409', 'J1431-5740', 'J1455-3330', 'J1514-4946', 'J1529-3828',
        'J1543-5149', 'J1545-4550', 'J1600-3053', 'J1618-3921', 'J1622-6617',
        'J1623-6936', 'J1625-0021', 'J1630+3734', 'J1640+2224',
        'J1643-1224', 'J1653-2054', 'J1708-3506', 'J1709+2313', 'J1711-4322',
        'J1713+0747', 'J1732-5049', 'J1738+0333', 'J1741+1351', 'J1745-0952',
        'J1751-2857', 'J1803-2712', 'J1804-2717', 'J1806+2819',
        'J1810-2005', 'J1811-0624', 'J1811-2405', 'J1813-0402', 'J1813-2621',
        'J1822-0848', 'J1823-3543', 'J1824+1014', 'J1824-0621', 'J1825-0319',
        'J1828+0625', 'J1835-0114', 'J1837-0822', 'J1840-0643', 'J1841+0130',
        'J1844+0115', 'J1850+0124', 'J1853+1303', 'J1855-1436', 'J1857+0943',
        'J1858-2216', 'J1900+0308', 'J1901+0300', 'J1902-5105', 'J1903-7051',
        'J1904+0412', 'J1906+0055', 'J1908+0128', 'J1909-3744', 'J1910+1256',
        'J1911-1114', 'J1912-0952', 'J1913+0618', 'J1918-0642', 'J1921+0137',
        'J1921+1929', 'J1929+0132', 'J1930+2441', 'J1932+1500', 'J1935+1726',
        'J1937+1658', 'J1938+2012', 'J1946+3417', 'J1950+2414', 'J1955+2908',
        'J2001+0701', 'J2006+0148', 'J2010+3051', 'J2015+0756', 'J2016+1948',
        'J2017+0603', 'J2019+2425', 'J2022+2534', 'J2033+1734', 'J2039-3616',
        'J2042+0246', 'J2043+1711', 'J2129-5721', 'J2204+2700', 'J2229+2643',
        'J2234+0611', 'J2236-5527', 'J2302+4442', 'J2317+1439', 'J2355+0051'
    ]

    gc_systems = [ # NS_WD(He) in globular clusters [gc] -- skew e-P more frequently due to additional gravitational field disturbances
        'J0024-7204H', 'J0024-7204Q', 'J0024-7204S', 'J0024-7204T', 'J0024-7204U',
        'J0024-7204Y', 'J0514-4002H', 'J0514-4002I', 'J1342+2822D', 'J1518+0204B',
        'J1518+0204C', 'J1518+0204D', 'J1518+0204E', 'J1623-2631',
        'J1641+3627B', 'J1641+3627D', 'J1641+3627F', 'J1701-3006A', 'J1701-3006B', 
        'J1701-3006D', 'J1701-3006G', 'J1748-2446am',
        'J1748-2446E', 'J1748-2446I', 'J1748-2446W', 'J1748-2446Y', 'J1824-2452I',
        'J1835-3259B', 'J1910-5959A'
    ]

    name = df['PSRJ']
    ecc = df['ECC'] # dimentionless
    medmass = df['MEDMASS'] # in solar masses. The mass of the companion star when i=60 deg
    minmass = df['MINMASS'] # in solar masses. The mass of the companion star when i=90 deg
    # we will add maxmass as the higher WD mass limit of 1.35 SM below
    pb = df['PB'] # in days
    age_i = df['AGE_I'] # in years

    mask = (ecc > 0) & (medmass > 0) & (pb > 0) & (age_i > 0) & (minmass > 0) # Select only double neutron star systems (BINCOMP starts with 'NS') and non-null eccentricity
    name_m = name[mask]
    ecc_m = ecc[mask]
    medmass_m = medmass[mask]
    pb_m = pb[mask]
    age_i_m = age_i[mask]
    minmass_m = minmass[mask]
    maxmass_m = np.zeros_like(minmass_m) # same matrix as medmass
    maxmass_m += 1.44 # Chandrasekhar limit 

    # pre-circularization values
    ecc_m_corr, a_m_corr, ecc_m_corr_min, a_m_corr_min, ecc_m_corr_max, a_m_corr_max = a0_retrieval(name_m, ecc_m, medmass_m, minmass_m, maxmass_m, pb_m, age_i_m, print_dt) # correct the eccentricity to the intiial post explosion ones
    
    # plotting
    plt.figure(figsize=(9.0, 6.1))

    name_m_values = name_m.values 
    mask_gc = np.array([name in gc_systems for name in name_m_values])
    mask_gf = np.array([name in gf_systems for name in name_m_values])
    
    def plot_population(mask, color, label_prefix, add_legend=True): # linked func to avoid writing the same twice
        x = pb_m.values[mask]
        y_obs = ecc_m.values[mask]
        y_min = ecc_m_corr_min[mask]
        y_med = ecc_m_corr[mask]
        y_max = ecc_m_corr_max[mask]

        # a thin line marks full range (min -> max)
        plt.vlines(x, y_min, y_max, color=color, linewidth=1.0, alpha=0.4)
        
        # underscores to mark min and max values
        plt.scatter(x, y_min, marker='_', color=color, s=30, alpha=0.6)
        plt.scatter(x, y_max, marker='_', color=color, s=30, alpha=0.6)

        # to mark 50% of possible systems, a thicker line goes from min (i = 90 deg) to median (i = 60 deg)
        label1 = f'{label_prefix} 50% Prob. Range ($i \in [60^\circ, 90^\circ]$)' if add_legend else ''
        plt.vlines(x, y_min, y_med, color=color, linewidth=2.0, alpha=0.7, label=label1)

        # naming and dots
        marker_color = 'cyan' if label_prefix=='GF' else 'orange' # cyan & orange are nice accent colors on blue & red, med ecc is thus highlighted
        label3 = f'{label_prefix} Median Initial' if add_legend else ''
        plt.scatter(x, y_med, marker='+', color=marker_color, s=30, zorder=10, label=label3)
        label2 = f'{label_prefix} Current Observed' if add_legend else ''
        plt.scatter(x, y_obs, marker='*', color=color, s = 20, alpha=0.8, zorder=10, label=label2)

    # gf systems
    if np.any(mask_gf):
        plot_population(mask_gf, 'blue', 'GF', add_legend=True)

    # gc systems
    if np.any(mask_gc):
        plot_population(mask_gc, 'red', 'GC', add_legend=True)
    
    # scale selector
    plt.xscale('log') if plot_type == 'log' else 'linear'
    plt.yscale('log') if plot_type == 'log' else 'linear'
    
    # Earth ecc
    xmin, xmax = plt.xlim()
    plt.hlines(0.0167, xmin, xmax, color='black', linestyles='--', label='Earth Eccentricity')
    plt.xlim(xmin, xmax)

    # line fitting for gc systems (uses current eccentricities for fitting)
    x_gf = pb_m.values[mask_gf]
    y_gf_med = ecc_m_corr[mask_gf]
    
    slope, intercept = fit_gc(x_gf, y_gf_med)
    if ((not np.isnan(slope)) and (not np.isnan(intercept))):
        x_f = np.linspace(min(x_gf), max(x_gf), 100)
        y_f = (10**intercept) * (x_f**slope)
        plt.plot(x_f, y_f, color='green', linestyle=':', linewidth=0.75)

        print('\n\nGC systems log fit parameters:')
        print(f'Log line: Slope is {slope}, Intercept is {intercept}')
        print(f'function formula: ecc = {10**intercept} * (P_b)^{slope}')
    

    plt.xlabel('Orbital Period (days)', fontsize=font)
    plt.ylabel('Eccentricity', fontsize=font)
    plt.title('e-P diagram for Initial vs. Current Eccentricities for NS-WD(He) Systems', fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # legend
    legend_elements = [
        # coloring of gf and gc
        Line2D([0], [0], color='blue', lw=1, label='Galactic Field systems (GF)'),
        Line2D([0], [0], color='red', lw=1, label='Globular Cluster systems (GC)'),
        
        # data symbols
        Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=8, label='Current Observed ecc'),
        Line2D([0], [0], marker='+', color='gray', linestyle='None', markersize=6, label='Median Initial ecc'),
        Line2D([0], [0], color='black', lw=2, alpha=0.7, label=r'50% prob. of incl. ($i \in [60^\circ, 90^\circ]$)'),
        
        # earth ref
        Line2D([0], [0], color='black', linestyle='--', label='Earth Eccentricity'),

        # line fit
        Line2D([0], [0], color='green', linestyle=':', lw=1.0, label='GC systems log_line fit')
    ]
    
    # text scaling
    plt.legend(handles=legend_elements, loc='lower right', fontsize=font, framealpha=0.9)

    if grid: plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
    #plt.savefig('NS-WD(He)_e-P_diagram.png', bbox_inches='tight', dpi = 2000)
    
    # exit frame
    if len(name_m) == 0 or len(ecc_m_corr) == 0:
        print("Warning: No NS binary systems were found")
        
    df_out = pd.DataFrame({
        'PSRJ': name_m.values,
        'ECC': ecc_m_corr
    })
    return df_out

# ---- line fitting for gc systems ----
def fit_gc(pb, ecc):
    m = (pb > 0) & (ecc > 0) & (ecc < 1)
    if m.sum() < 2: return np.Nan(), np.Nan()
    pb_log = np.log10(pb[m]) # we have a linear slope only in log (id est a power law)
    ecc_log = np.log10(ecc[m])
    der, inter = np.polyfit(pb_log, ecc_log, 1) # we get slope and intercept
    return der, inter

# ---- WD mass function ----
def wd_mass_function():
    print(np.NaN())

def companion_mass_function():
    print(np.NaN())

# ---- simulate mass loss ----

def simulate_mass_loss(
) -> pd.DataFrame:
    """
    This function reiterates WD mass loss over period of tau, four times a period P0"""

    # initial parameters for our specific case:
    bound = True
    tau = 2 * 365 * 24 * 3600 # median duration of terminal burst phase of white dwarfs (sec)
    P0 = 50 * 24 * 3600 # initial orbital period of NS-WD (sec)
    n_steps = 4 * int(tau / P0) # number of integration steps to take
    e_ini = 0.05 # we don't expect NS-WD(H) to have high eccentricities
    f_ini = 0.0 # start at zero true anomaly for the ease of mind

    m2 = 1.4 # mass of the neutron star {SM}
    m1_pre = 1.4 # mass of the WD progenitor before the final mass loss {SM}
    m1_post = 0.6 # mass of the WD after the final mass loss {SM}

    v_kick_net_mag = 5.0 # overall terminal burst kick magnitude {km/s}
    # ----------{end of IC}----------
    v_kick_mag_step = v_kick_net_mag / n_steps # small kick magnitude per step {km/s}

    dm_total = m1_pre - m1_post
    dm_step = dm_total / n_steps

    a0 = ((G_AU * (m1_pre + m2) * (P0 / 86400)**2) / (4 * pi**2)) ** (1.0/3.0) # initial semi-major axis {AU}

    # the calculation loop
    a0_last = a0
    e_last = e_ini
    f_last = f_ini
    m1_last = m1_pre
    for i in range(n_steps):
        if bound == True:
            m1_current_step_post = m1_last - dm_step

            bound, a_curr, e_curr = single_mass_loss(
                m1_last,
                m1_current_step_post,
                m2,
                a0_last,
                v_kick_mag_step,
                f_last,
                e_last
            )

            a0_last = a_curr
            e_last = e_curr
            f_last = (f_last + 0.5 * pi) % (2 * pi) # advance by pi/2
            m1_last = m1_current_step_post

        if bound == False:
            print(f"System became unbound at step {i+1} out of {n_steps}")
            break
    
    return ((a0, a0_last), (e_ini, e_last), bound)

def single_mass_loss(
    m1_pre,          # mass of the primary star before mass loss{SM}
    m1_post,         # mass of the primary star after mass loss {SM}
    m2,              # mass of the companion star {SM}
    a0,              # initial semi-major axis {AU}
    v_kick_mag,      # kick velocity magnitude (to be rotated) {km/s}
    f_ini,           # true anomaly {radians}
    e_ini            # eccentricity
):
    """
    Simulates a single step of mass loss and kick for a binary in an elliptical orbit.
    """
    M0 = m1_pre + m2 # initial total mass {M_sun}
    Mt = m1_post + m2 # total mass after step {M_sun}
    v_c = np.sqrt(G_AU * M0 / a0) * AU_PER_DAY_KMS # reference circular orbital speed {km/s}

    # Position vector in perifocal frame (f=0 is +X)
    r_mag = a0 * (1 - e_ini**2) / (1 + e_ini * np.cos(f_ini)) # instantaneous separation {AU}
    r_x = r_mag * np.cos(f_ini)
    r_y = r_mag * np.sin(f_ini)
    r_vec = np.array([r_x, r_y, 0.0]) # position vector {AU}

    # Orbital velocity vector in perifocal frame {km/s}
    v_orb_x = -v_c * np.sin(f_ini) / np.sqrt(1 - e_ini**2)
    v_orb_y = v_c * (np.cos(f_ini) + e_ini) / np.sqrt(1 - e_ini**2)
    v_orb_z = 0.0
    v_orb_vec = np.array([v_orb_x, v_orb_y, v_orb_z])
    
    # Rotate kick to be prograde (in direction of v_orb_vec)
    v_orb_mag = np.linalg.norm(v_orb_vec)
    if v_orb_mag > 1e-9: # /0
        v_kick_vec = v_kick_mag * (v_orb_vec / v_orb_mag) # mag * unti vector gives a directed vector. Set colinear to v_orb
    else:
        v_kick_vec = np.array([0.0, 0.0, 0.0]) # No velocity to be prograde to

    # Astrodynamics calculations
    v_fin_vec = v_orb_vec + v_kick_vec # final velocity vector {km/s}
    v_fin_mag2 = np.sum(v_fin_vec**2) # final speed squared {km^2/s^2}
    v_fin_mag2_AU = v_fin_mag2 / AU_PER_DAY_KMS**2 # convert to AU^2/day^2

    mu_post = G_AU * Mt # gravitational parameter after step {AU^3/day^2}
    specific_E = 0.5 * v_fin_mag2_AU - mu_post / r_mag  # specific orbital energy {AU^2/day^2}
    bound = specific_E < 0.0 # bound if specific energy is negative

    a_final = 0
    e_final = 0

    if bound: # process only bound systems
        a_final = -mu_post / (2.0 * specific_E) # semi-major axis post step {AU}
        
        v_fin_AU = v_fin_vec / AU_PER_DAY_KMS # final velocity {AU/day}
        h_vec = np.cross(r_vec, v_fin_AU) # specific angular momentum vector {AU^2/day}
        h2 = np.sum(h_vec**2) # squared magnitude {AU^4/day^2}
        
        e_sq = 1.0 - h2 / (mu_post * a_final) # eccentricity squared
        e_sq = max(0.0, e_sq) # ensure non-negative
        e_final = np.sqrt(e_sq) # final eccentricity

    return bound, a_final, e_final

# ---- execution ----

if __name__ == "__main__":

    # Step I -- plotting inclination and circularization corrections in e-P diagram
    # Step II -- line fitting for log gc systems
    load_dns_catalogue(
        plot_type='log', # 'log' or 'linear'. "linear" does not stick well here though
        path='', # if not in the same directory, should start from CD to the folder. Otherwise, leave ''.
        cache_file='dns_He_catalogue.csv',
        print_dt = False, # prints auxiliary data table {outdated}
        grid = False, # grid on the graph (bool)
        font = 'medium' # font size str
    )

    # Step III -- simulating WD assymmetric mass loss {one example as of now}
    data = simulate_mass_loss()

    print('\n\n')
    print ('Simulation Data:')
    print('\n')
    print('Initial semi-major axis (AU), Final semi-major axis (AU): ', data[0])
    print('Initial eccentricity, Final eccentricity: ', data[1])

    if data[2]: print('The system remained bound after mass loss.')
    else: print('The system became unbound after mass loss.')


    print('\nAll simulations successful.\n') # flag that no errors occurred and files are re-generated
