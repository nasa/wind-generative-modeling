from wdm_src.utils import get_vector_cols
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.2)


if __name__ == '__main__':

    real = pd.read_csv('data/combined_macro_micro_data.csv')
    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    real = real[u_cols + v_cols]
    real.dropna(inplace=True)

    ddpm = pd.read_csv(
        'scripts/postprocessing/unconditional/ddpm_unconditional.csv')
    fm = pd.read_csv(
        'scripts/postprocessing/unconditional/fm_unconditional.csv')
    gmm = pd.read_csv(
        'scripts/postprocessing/unconditional/gmm_unconditional.csv')

    '''
    The following three plots make each cell of Figure 5. in the paper.
    '''

    '''
    GMM bivariate plot.
    '''
    u_real = real[get_vector_cols('u')].values.mean(1)
    v_real = real[get_vector_cols('v')].values.mean(1)

    u_pred = gmm[get_vector_cols('u')].values.mean(1)
    v_pred = gmm[get_vector_cols('v')].values.mean(1)

    g = sns.JointGrid(xlim=(-10.0, 20.0), ylim=(-20.0, 20.0))
    g.figure.suptitle('GMM')
    sns.scatterplot(x=u_pred, y=v_pred, color='black',
                    s=5, alpha=0.4, ax=g.ax_joint)

    sns.kdeplot(x=u_pred, fill=True, ax=g.ax_marg_x, label='Generated')
    sns.kdeplot(y=v_pred, fill=True, ax=g.ax_marg_y)
    sns.kdeplot(x=u_real, fill=False, ax=g.ax_marg_x,
                linestyle='--', color='red', label='Real')
    sns.kdeplot(y=v_real, fill=False, ax=g.ax_marg_y,
                linestyle='--', color='red')

    g.ax_marg_x.legend()
    g.set_axis_labels('U Velocity (m/s)', 'V Velocity (m/s)')
    g.figure.tight_layout()
    g.figure.savefig('bivariate_gmm.png', dpi=600)

    '''
    DDPM bivariate plot
    '''
    u_real = real[get_vector_cols('u')].values.mean(1)
    v_real = real[get_vector_cols('v')].values.mean(1)

    u_pred = ddpm[get_vector_cols('u')].values.mean(1)
    v_pred = ddpm[get_vector_cols('v')].values.mean(1)

    g = sns.JointGrid(xlim=(-10.0, 20.0), ylim=(-20.0, 20.0))
    g.figure.suptitle('DDPM')
    sns.scatterplot(x=u_pred, y=v_pred, color='black',
                    s=5, alpha=0.4, ax=g.ax_joint)

    sns.kdeplot(x=u_pred, fill=True, ax=g.ax_marg_x, label='Generated')
    sns.kdeplot(y=v_pred, fill=True, ax=g.ax_marg_y)
    sns.kdeplot(x=u_real, fill=False, ax=g.ax_marg_x,
                linestyle='--', color='red', label='Real')
    sns.kdeplot(y=v_real, fill=False, ax=g.ax_marg_y,
                linestyle='--', color='red')

    g.ax_marg_x.legend()
    g.set_axis_labels('U Velocity (m/s)', 'V Velocity (m/s)')
    g.figure.tight_layout()
    g.figure.savefig('bivariate_ddpm.png', dpi=600)

    '''
    FM bivariate plot
    '''
    u_real = real[get_vector_cols('u')].values.mean(1)
    v_real = real[get_vector_cols('v')].values.mean(1)

    u_pred = fm[get_vector_cols('u')].values.mean(1)
    v_pred = fm[get_vector_cols('v')].values.mean(1)

    g = sns.JointGrid(xlim=(-10.0, 20.0), ylim=(-20.0, 20.0))
    g.figure.suptitle('FM')
    sns.scatterplot(x=u_pred, y=v_pred, color='black',
                    s=5, alpha=0.4, ax=g.ax_joint)

    sns.kdeplot(x=u_pred, fill=True, ax=g.ax_marg_x, label='Generated')
    sns.kdeplot(y=v_pred, fill=True, ax=g.ax_marg_y)
    sns.kdeplot(x=u_real, fill=False, ax=g.ax_marg_x,
                linestyle='--', color='red', label='Real')
    sns.kdeplot(y=v_real, fill=False, ax=g.ax_marg_y,
                linestyle='--', color='red')

    g.ax_marg_x.legend()
    g.set_axis_labels('U Velocity (m/s)', 'V Velocity (m/s)')
    g.figure.tight_layout()
    g.figure.savefig('bivariate_fm.png', dpi=600)

    '''
    Finally stitch all three bivariate plots together.
    '''
    sns.set_theme(font_scale=1.2)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(plt.imread('bivariate_gmm.png'))
    ax[0].axis('off')
    ax[1].imshow(plt.imread('bivariate_ddpm.png'))
    ax[1].axis('off')
    ax[2].imshow(plt.imread('bivariate_fm.png'))
    ax[2].axis('off')

    fig.tight_layout()
    fig.savefig('stitch.png', dpi=600)
