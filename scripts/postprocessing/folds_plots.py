from wdm_src import plots
from wdm_src.utils import get_vector_cols
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.2)


def kl_divergence(x1: np.array, x2: np.array):
    p_kde = gaussian_kde(x1)
    q_kde = gaussian_kde(x2)

    p = p_kde.pdf(x1)
    q = q_kde.pdf(x1)
    return np.log(p / q).mean()


if __name__ == '__main__':
    '''
    This figure is comparing a cherrypicked combination of macroweather conditions (not seen during training).
    It demonstrates the performance of each model to generate samples that match the real samples.
    '''
    real = pd.read_csv('data/combined_macro_micro_data.csv')

    speed_folds = ['(-0.001, 2.235]', '(2.235, 5.364]',
                   '(5.364, 8.047]', '(8.047, 15.646]']
    direction_folds = ['SW', 'W', 'WNW', 'WSW']
    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    real = real[u_cols + v_cols + ['macro_ws_str', 'macro_wd_str']]
    real.dropna(inplace=True)

    withhold_speed = speed_folds[2]
    withhold_direction = direction_folds[0]
    real_fold = real[(real['macro_ws_str'] == withhold_speed) &
                     (real['macro_wd_str'] == withhold_direction)]

    ddpm = pd.read_csv(
        f'scripts/postprocessing/folds/ddpm/{withhold_speed}-{withhold_direction}.csv')
    ddpm = ddpm[(ddpm['macro_ws_str'] == withhold_speed) &
                (ddpm['macro_wd_str'] == withhold_direction)]

    fm = pd.read_csv(
        f'scripts/postprocessing/folds/fm/{withhold_speed}-{withhold_direction}.csv')
    fm = fm[(fm['macro_ws_str'] == withhold_speed) &
            (fm['macro_wd_str'] == withhold_direction)]

    gmm = pd.read_csv(
        f'scripts/postprocessing/folds/gmm/{withhold_direction}_{withhold_speed}.csv')

    u = real_fold[u_cols].values.mean(1)
    v = real_fold[v_cols].values.mean(1)
    uv = np.stack([u, v])

    u_ddpm = ddpm[u_cols].values.mean(1)
    v_ddpm = ddpm[v_cols].values.mean(1)
    uv_ddpm = np.stack([u_ddpm, v_ddpm])

    u_fm = fm[u_cols].values.mean(1)
    v_fm = fm[v_cols].values.mean(1)
    uv_fm = np.stack([u_fm, v_fm])

    u_gmm = gmm[u_cols].values.mean(1)
    v_gmm = gmm[v_cols].values.mean(1)
    uv_gmm = np.stack([u_gmm, v_gmm])

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
    ax[0].set_ylabel('U Velocity (m/s)')
    ax[0].set_title('GMM')
    ax[0].set_xlim(-5.0, 25.0)
    ax[0].set_ylim(-5.0, 25.0)
    ax[0].scatter(u, v)
    ax[0].scatter(u_gmm, v_gmm, color='orange', alpha=0.5)
    ax[0].set_xlabel('V Velocity (m/s)')
    ax[0].text(15.0, 20.0, 'KLD = ' +
               str(round(kl_divergence(uv, uv_gmm) + kl_divergence(uv_gmm, uv), 2)))

    ax[1].set_title('DDPM')
    ax[1].set_xlim(-5.0, 25.0)
    ax[1].set_ylim(-5.0, 25.0)
    ax[1].scatter(u, v)
    ax[1].scatter(u_ddpm, v_ddpm, color='orange', alpha=0.5)
    ax[1].set_xlabel('V Velocity (m/s)')
    ax[1].text(15.0, 20.0, 'KLD = ' +
               str(round(kl_divergence(uv, uv_ddpm) + kl_divergence(uv_ddpm, uv), 2)))

    ax[2].set_title('FM')
    ax[2].set_xlim(-5.0, 25.0)
    ax[2].set_ylim(-5.0, 25.0)
    ax[2].scatter(u, v)
    ax[2].scatter(u_fm, v_fm, color='orange', alpha=0.5)
    ax[2].set_xlabel('V Velocity (m/s)')
    ax[2].text(15.0, 20.0, 'KLD = ' +
               str(round(kl_divergence(uv, uv_fm) + kl_divergence(uv_fm, uv), 2)))
    fig.tight_layout()
    fig.savefig('cherrypick.png', dpi=600)

    '''
    GMM on each fold.
    '''
    fig, ax = plt.subplots(len(speed_folds), len(
        direction_folds), figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Real vs GMM Samples on Each Fold')

    for i in range(len(speed_folds)):
        withhold_speed = speed_folds[i]
        ax[i, 0].set_ylabel(f'{withhold_speed}\nm/s')

        for j in range(len(direction_folds)):
            withhold_direction = direction_folds[j]
            if i == 0:
                ax[i, j].set_title(withhold_direction)
            if i == len(speed_folds)-1:
                ax[i, j].set_xlabel('m/s')

            real_fold = real[(real['macro_ws_str'] == withhold_speed) & (
                real['macro_wd_str'] == withhold_direction)]
            gmm = pd.read_csv(
                f'scripts/postprocessing/folds/gmm/{withhold_direction}_{withhold_speed}.csv')

            u = real_fold[u_cols].values.mean(1)
            v = real_fold[v_cols].values.mean(1)
            uv = np.stack([u, v])

            u_gmm = gmm[u_cols].values.mean(1)
            v_gmm = gmm[v_cols].values.mean(1)
            uv_gmm = np.stack([u_gmm, v_gmm])

            ax[i, j].set_xlim(-15.0, 25.0)
            ax[i, j].set_ylim(-15.0, 25.0)
            ax[i, j].scatter(u, v)
            ax[i, j].scatter(u_gmm, v_gmm, color='orange', alpha=0.5)

    fig.tight_layout()
    fig.savefig('GMM.png', dpi=600)

    '''
    DDPM on each fold.
    '''
    fig, ax = plt.subplots(len(speed_folds), len(
        direction_folds), figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Real vs DDPM Samples on Each Fold')
    for i in range(len(speed_folds)):
        withhold_speed = speed_folds[i]
        ax[i, 0].set_ylabel(f'{withhold_speed}\nm/s')

        for j in range(len(direction_folds)):
            withhold_direction = direction_folds[j]
            if i == 0:
                ax[i, j].set_title(withhold_direction)
            if i == len(speed_folds)-1:
                ax[i, j].set_xlabel('m/s')

            real_fold = real[(real['macro_ws_str'] == withhold_speed) & (
                real['macro_wd_str'] == withhold_direction)]

            ddpm = pd.read_csv(
                f'scripts/postprocessing/folds/ddpm/{withhold_speed}-{withhold_direction}.csv')
            ddpm = ddpm[(ddpm['macro_ws_str'] == withhold_speed) &
                        (ddpm['macro_wd_str'] == withhold_direction)]

            u = real_fold[u_cols].values.mean(1)
            v = real_fold[v_cols].values.mean(1)
            uv = np.stack([u, v])

            u_ddpm = ddpm[u_cols].values.mean(1)
            v_ddpm = ddpm[v_cols].values.mean(1)
            uv_ddpm = np.stack([u_ddpm, v_ddpm])

            ax[i, j].set_xlim(-15.0, 25.0)
            ax[i, j].set_ylim(-15.0, 25.0)
            ax[i, j].scatter(u, v)
            ax[i, j].scatter(u_ddpm, v_ddpm, color='orange', alpha=0.5)

    fig.tight_layout()
    fig.savefig('DDPM.png', dpi=600)

    '''
    FM on each fold.
    '''
    fig, ax = plt.subplots(len(speed_folds), len(
        direction_folds), figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Real vs FM Samples on Each Fold')
    for i in range(len(speed_folds)):
        withhold_speed = speed_folds[i]
        ax[i, 0].set_ylabel(f'{withhold_speed}\nm/s')

        for j in range(len(direction_folds)):
            withhold_direction = direction_folds[j]
            if i == 0:
                ax[i, j].set_title(withhold_direction)
            if i == len(speed_folds)-1:
                ax[i, j].set_xlabel('m/s')

            real_fold = real[(real['macro_ws_str'] == withhold_speed) & (
                real['macro_wd_str'] == withhold_direction)]

            fm = pd.read_csv(
                f'scripts/postprocessing/folds/fm/{withhold_speed}-{withhold_direction}.csv')
            fm = fm[(fm['macro_ws_str'] == withhold_speed) &
                    (fm['macro_wd_str'] == withhold_direction)]

            u = real_fold[u_cols].values.mean(1)
            v = real_fold[v_cols].values.mean(1)
            uv = np.stack([u, v])

            u_fm = fm[u_cols].values.mean(1)
            v_fm = fm[v_cols].values.mean(1)
            uv_fm = np.stack([u_fm, v_fm])

            ax[i, j].set_xlim(-15.0, 25.0)
            ax[i, j].set_ylim(-15.0, 25.0)
            ax[i, j].scatter(u, v)
            ax[i, j].scatter(u_fm, v_fm, color='orange', alpha=0.5)

    fig.tight_layout()
    fig.savefig('FM.png', dpi=600)
