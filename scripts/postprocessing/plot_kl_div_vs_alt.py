from wdm_src.utils import ALTITUDES, get_vector_cols
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style='dark')
sns.set_style('darkgrid')

NUM_ALT = len(ALTITUDES)


def get_kl_div_vs_altitude(measurement_data, generated_data, N=10000):
    u_meas = measurement_data[:, :NUM_ALT]
    v_meas = measurement_data[:, NUM_ALT:]
    u_gen = generated_data[:, :NUM_ALT]
    v_gen = generated_data[:, NUM_ALT:]
    kl_divs = np.zeros(NUM_ALT)
    for i in range(NUM_ALT):
        print("\tAlt. # {} / {}".format(i+1, NUM_ALT))
        data_kde = gaussian_kde(np.vstack((u_meas[:, i], v_meas[:, i])))
        gen_kde = gaussian_kde(np.vstack((u_gen[:, i], v_gen[:, i])))

        # evaluate symmetric KL divergence
        sym_kl_div = 0
        points = data_kde.resample(N)
        p = data_kde.pdf(points)
        q = gen_kde.pdf(points)
        p_over_q = p / q
        p_over_q = p_over_q[np.where(~np.isinf(p_over_q))]
        kl_divergence = np.log(p_over_q).mean()
        sym_kl_div += kl_divergence

        points = gen_kde.resample(N)
        p = gen_kde.pdf(points)
        q = data_kde.pdf(points)
        kl_divergence = np.log(p / q).mean()
        sym_kl_div += kl_divergence
        kl_divs[i] = sym_kl_div
    return kl_divs


if __name__ == '__main__':
    '''
    Computes the KL divergence of the generated distributions using DDPM, FM, and GMM vs. the measurement distribution for each altitude to reproduce Figure 7 in the paper. This script takes a few minutes to run.
    '''
    np.random.seed(0)
    N = 10000       # num. samples from KDEs to compute KL divergence

    # Load measurement data; pull out relevant columsn; drop nan data
    measurement_df = pd.read_csv('data/combined_macro_micro_data.csv')
    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    measurement_df = measurement_df[u_cols + v_cols]
    measurement_df.dropna(inplace=True)
    measurement_data = measurement_df.values

    # last two cols are macro conds
    gmm_df = pd.read_csv(
        'scripts/postprocessing/unconditional/gmm_unconditional.csv')
    gmm_data = gmm_df.values[:, :-2]

    # read DDPM & FM samples
    ddpm_df = pd.read_csv(
        'scripts/postprocessing/unconditional/ddpm_unconditional.csv')
    fm_df = pd.read_csv(
        'scripts/postprocessing/unconditional/fm_unconditional.csv')
    fm_df = fm_df[u_cols + v_cols]
    ddpm_df = ddpm_df[u_cols + v_cols]
    fm_data = fm_df.values
    ddpm_data = ddpm_df.values

    print("Getting FM KL div")
    fm_kl_divs = get_kl_div_vs_altitude(measurement_data, fm_data, N)
    print("Getting DDPM KL div")
    ddpm_kl_divs = get_kl_div_vs_altitude(measurement_data, ddpm_data, N)
    print("Getting GMM KL div")
    gmm_kl_divs = get_kl_div_vs_altitude(measurement_data, gmm_data, N)

    plt.figure()
    plt.plot(ALTITUDES, gmm_kl_divs, '-s', label='GMM')
    plt.plot(ALTITUDES, ddpm_kl_divs, '-^', label='DDPM')
    plt.plot(ALTITUDES, fm_kl_divs, '-o', label='FM')
    plt.xlabel('Altitude (m)')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.savefig('kl_div_vs_altitude_comparison.pdf')
    plt.show()
