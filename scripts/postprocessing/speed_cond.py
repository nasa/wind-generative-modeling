from wdm_src.utils import get_vector_cols, ALTITUDES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.2)

if __name__ == '__main__':

    real = pd.read_csv('data/combined_macro_micro_data.csv')
    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    ws_cols = get_vector_cols('ws')
    real = real[u_cols + v_cols + ws_cols + ['macro_ws_str', 'macro_wd_str']]
    real.dropna(inplace=True)
    ddpm = pd.read_csv('scripts/postprocessing/speed_cond/ddpm.csv')
    fm = pd.read_csv('scripts/postprocessing/speed_cond/fm.csv')

    # load in each segment of the gmm dataset seperately.
    # combine into one dataset once the irrelevant columns have been stripped off.
    gmm_1 = pd.read_csv(
        'scripts/postprocessing/speed_cond/gmm_generated_data_uv_2.23_6542.csv', index_col=[0])
    gmm_1['macro_ws_str'] = '(-0.001, 2.235]'

    gmm_2 = pd.read_csv(
        'scripts/postprocessing/speed_cond/gmm_generated_data_uv_5.36_6542.csv', index_col=[0])
    gmm_2['macro_ws_str'] = '(2.235, 5.364]'

    gmm_3 = pd.read_csv(
        'scripts/postprocessing/speed_cond/gmm_generated_data_uv_8.05_6542.csv', index_col=[0])
    gmm_3['macro_ws_str'] = '(5.364, 8.047]'

    gmm_4 = pd.read_csv(
        'scripts/postprocessing/speed_cond/gmm_generated_data_uv_15.65_6542.csv', index_col=[0])
    gmm_4['macro_ws_str'] = '(8.047, 15.646]'
    gmm = pd.concat([gmm_1, gmm_2, gmm_3, gmm_4], ignore_index=True, axis=0)

    gmm[ws_cols] = (gmm[u_cols].values ** 2 + gmm[v_cols].values**2) ** (0.5)

    mean_real = real.groupby('macro_ws_str')[ws_cols].mean()
    std_real = real.groupby('macro_ws_str')[ws_cols].std()
    speeds = mean_real.index.values

    mean_ddpm = ddpm.groupby('macro_ws_str')[ws_cols].mean()
    std_ddpm = ddpm.groupby('macro_ws_str')[ws_cols].std()

    mean_fm = fm.groupby('macro_ws_str')[ws_cols].mean()
    std_fm = fm.groupby('macro_ws_str')[ws_cols].std()

    mean_gmm = gmm.groupby('macro_ws_str')[ws_cols].mean()
    std_gmm = gmm.groupby('macro_ws_str')[ws_cols].std()

    models = ['GMM', 'DDPM', 'FM']
    means = [mean_gmm, mean_ddpm, mean_fm]
    stds = [std_gmm, std_ddpm, std_fm]

    '''
    Makes Figure 8. in the paper
    '''
    fig, ax = plt.subplots(3, len(speeds),
                           figsize=(16, 16),
                           sharex=True,
                           sharey=True
                           )

    for (i, macro_speed) in enumerate(speeds):
        l, h = macro_speed.split(',')
        l = round(abs(float(l.replace('(', ''))), 2)
        h = round(abs(float(h.replace(']', ''))), 2)
        title = f'Macroweather Speed: ({l}-{h}) (m/s)'

        ax[0, i].set_title(title)
        ax[-1, i].set_xlabel('Altitude (m)')

    for (j, m) in enumerate(models):
        ax[j, 0].text(-0.2, 0.5, m, transform=ax[j, 0].transAxes,
                      rotation=90, va='center', ha='center', weight='bold')

    for (j, m) in enumerate(models):
        ax[j, 0].set_ylabel('Microweather Speed (m/s)')

        for (i, macro_speed) in enumerate(speeds):
            mean_real_speed = mean_real.values[i, :]
            std_real_speed = std_real.values[i, :]

            mean_pred_speed = means[j].values[i, :]
            std_pred_speed = stds[j].values[i, :]

            ax[j, i].plot(ALTITUDES, mean_real_speed,
                          label='Real', color='blue')
            ax[j, i].fill_between(ALTITUDES, mean_real_speed - std_real_speed,
                                  mean_real_speed + std_real_speed, alpha=0.2, color='blue')

            ax[j, i].plot(ALTITUDES, mean_pred_speed,
                          label='Generated', color='orange')
            ax[j, i].fill_between(ALTITUDES, mean_pred_speed - std_pred_speed,
                                  mean_pred_speed + std_pred_speed, alpha=0.2, color='orange')
            ax[j, i].legend()
            ax[j, i].set_ylim(0, 17)

    fig.tight_layout()
    fig.savefig('model_macro_micro_speed.png', dpi=600)
