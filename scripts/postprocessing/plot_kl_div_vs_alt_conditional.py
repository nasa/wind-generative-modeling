from wdm_src.utils import get_vector_cols, ALTITUDES
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style='dark')
sns.set_style('darkgrid')

NUM_ALT = len(ALTITUDES)


def get_conditional_measurement_df(measurement_df, speed):
    wind_df = measurement_df.copy(deep=True)
    wind_df = wind_df[wind_df.loc[:, 'macro_ws_str'] == speed]
    vel_cols = get_vector_cols('u') + get_vector_cols('v')
    wind_df = wind_df[vel_cols]
    wind_df.dropna(axis=1, how='all', inplace=True)
    wind_df.dropna(inplace=True)
    return wind_df


def get_conditional_dgm_df(df, speed):
    wind_df = df.copy(deep=True)
    wind_df = wind_df[wind_df.loc[:, 'macro_ws_str'] == speed]
    vel_cols = get_vector_cols('u') + get_vector_cols('v')
    wind_df = wind_df[vel_cols]
    return wind_df


def get_gmm_df(gmm_file):
    gmm_df = pd.read_csv(gmm_file)
    vel_cols = get_vector_cols('u') + get_vector_cols('v')
    return gmm_df[vel_cols]


def get_kl_div_vs_altitude(measurement_data, generated_data, N=10000):
    u_meas = measurement_data[:, :NUM_ALT]
    v_meas = measurement_data[:, NUM_ALT:]
    u_gen = generated_data[:, :NUM_ALT]
    v_gen = generated_data[:, NUM_ALT:]
    kl_divs = np.zeros(NUM_ALT)
    for i in range(NUM_ALT):
        print("Alt. # {} / {}".format(i+1, NUM_ALT))
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
    Computes the KL divergence of the generated distributions using DDPM, FM, and GMM vs. the measurement distribution for each altitude & macroweather wind condition to reproduce Figure 9 in the paper. This script takes a few minutes to run.
    '''
    np.random.seed(0)
    N = 10000
    MACRO_SPEEDS = ['(-0.001, 2.235]', '(2.235, 5.364]', '(5.364, 8.047]',
                    '(8.047, 15.646]']
    gmm_files = ['scripts/postprocessing/speed_cond/gmm_generated_data_uv_{}_6542.csv'.format(suffix)
                 for suffix in ['2.23', '5.36', '8.05', '15.65']]
    measurement_df = pd.read_csv('data/combined_macro_micro_data.csv')
    ddpm_df_all = pd.read_csv('scripts/postprocessing/speed_cond/ddpm.csv')
    fm_df_all = pd.read_csv('scripts/postprocessing/speed_cond/fm.csv')

    fig, ax = plt.subplots(1, len(MACRO_SPEEDS), figsize=(15, 5), sharey=True)

    for i, macro_speed in enumerate(MACRO_SPEEDS):

        print("macro speed", macro_speed)
        true_df = get_conditional_measurement_df(measurement_df, macro_speed)
        fm_df = get_conditional_dgm_df(fm_df_all,  macro_speed)
        ddpm_df = get_conditional_dgm_df(ddpm_df_all,  macro_speed)
        gmm_df = get_gmm_df(gmm_files[i])

        print("getting FM KL div")
        fm_kl_divs = get_kl_div_vs_altitude(true_df.values, fm_df.values, N)
        print("getting DDPM KL div")
        ddpm_kl_divs = get_kl_div_vs_altitude(true_df.values, ddpm_df.values,
                                              N)
        print("getting GMM KL div")
        gmm_kl_divs = get_kl_div_vs_altitude(true_df.values, gmm_df.values, N)

        l, h = macro_speed.split(',')
        l = round(abs(float(l.replace('(', ''))), 2)
        h = round(abs(float(h.replace(']', ''))), 2)
        title = f'Macroweather Speed: ({l}-{h}) (m/s)'

        ax[i].set_title(title)
        ax[i].set_xlabel('Altitude (m)')
        ax[i].set_ylabel('KL Divergence')
        ax[i].plot(ALTITUDES, gmm_kl_divs, '-s',
                   label='GMM')
        ax[i].plot(ALTITUDES, ddpm_kl_divs, '-^',
                   label='DDPM')
        ax[i].plot(ALTITUDES, fm_kl_divs, '-o',
                   label='FM')
        ax[i].legend()

    fig.tight_layout()
    fig.savefig('conditional_kl_div_vs_altitude.pdf')
    plt.show()
