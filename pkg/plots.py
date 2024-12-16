from pathlib import Path
from datetime import datetime

import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style = 'dark')
import numpy as np

from wdm_src.utils import get_vector_cols, ALTITUDES, DIRS

def wind_speed_macro_condition_line_plot(data: pd.DataFrame, output_dir: str = '', prefix: str = '', name: str = 'wind_speed_condition_line_plot.png'):
    # ws = data.groupby('macro_conditions')[get_vector_cols('ws')].mean()
    ws = data.groupby('is_fair')[get_vector_cols('ws')].mean()
    ws_values = ws.values
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    ax.set_title('Wind Speed vs Altitude by Weather Condition')
    for i in range(ws_values.shape[0]):
        ax.plot(ws_values[i, :], ALTITUDES, label = ws.index[i])
    ax.set_xlabel('m/s')
    ax.set_ylabel('m')
    ax.legend()
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_speed_macro_condition_bar_plot(data: pd.DataFrame, output_dir: str = '', prefix: str = '', name: str = 'wind_speed_condition_bar_plot.png'):
    ## bar chart of the counts of each condition
    macro_conditions_counts = data['macro_conditions'].value_counts()
    fig, ax = plt.subplots(1, 1, figsize = (10, 15))
    ax.set_title('Conditions Value Counts')
    ax.bar(macro_conditions_counts.index, macro_conditions_counts.values)
    ax.tick_params(axis = 'x', labelrotation = 45)
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_speed_macro_speed_plot(data: pd.DataFrame, bins: int = 4, output_dir: str = '', prefix: str = '', name: str = 'wind_speed_macro_speed_plot.png'):
    macro_ws_bins = pd.qcut(data['macro_ws'], bins) ## qcut instead of cut in order to bin into 'almost' equal size counts
    micro_ws = data[get_vector_cols('ws')].groupby(macro_ws_bins, observed = False).mean()
    micro_ws_values = micro_ws.values

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    for i in range(micro_ws_values.shape[0]): 
        ax[0].plot(micro_ws_values[i, :], ALTITUDES, label = micro_ws.index[i])

    ax[0].set_title('Micro Wind Speed vs Altitude Grouped by Macro Wind Speed')
    ax[0].set_xlabel('Wind Speed (m/s)')
    ax[0].set_ylabel('Altitude (m)')
    ax[0].legend()

    ax[1].set_title('Counts of Macro Wind Speed Bins')
    ax[1].bar(macro_ws_bins.value_counts(sort = False).index.astype(str), macro_ws_bins.value_counts(sort = False).values)
    ax[1].tick_params(axis = 'x', labelrotation = 45)
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_direction_macro_direction_bar_plot(data: pd.DataFrame, output_dir: str = '', prefix: str = '', name: str = 'wind_direction_bar_plot.png'):
    macro_wd_counts = data['macro_wd_str'].value_counts()
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    ax.set_title('Wind Direction Value Counts')
    ax.bar(macro_wd_counts.index, macro_wd_counts.values)
    ax.tick_params(axis = 'x', labelrotation = 45)
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_direction_macro_direction_polar_scatter_plot(data: pd.DataFrame, min_wind_speed: float = 0.0, output_dir: str = '', prefix: str = '', name: str = 'wind_direction_polar_scatter_plot.png'):

    fig, ax = plt.subplots(4, 4, subplot_kw={'projection': 'polar'}, figsize = (20, 25))
    for i, dir in enumerate(sorted(DIRS)):
        macro_direction = data[data['macro_wd_str'] == dir]
        macro_direction = macro_direction[macro_direction['macro_ws'] >= min_wind_speed]

        # angles = macro_direction[get_vector_cols('wd')].values.mean(axis=1) * np.pi / 180
        # speeds = macro_direction[get_vector_cols('ws')].values.mean(axis=1)

        u = macro_direction[get_vector_cols('u')].values
        v = macro_direction[get_vector_cols('v')].values

        speeds = np.sqrt(u ** 2 + v ** 2).mean(axis = 1)
        angles = np.arctan2(v, u).mean(axis = 1)

        y = i // 4 - 1
        x = i % 4 - 1
        ax[y, x].scatter(angles, speeds, alpha = 0.4)
        ax[y, x].set_title(dir)
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_confidence_signals_line_plot(data: pd.DataFrame, output_dir: str = '', prefix: str = '', name: str = 'wind_confidence_signals_line_plot.png'):
    
    fig, ax = plt.subplots(3, 1, figsize = (10, 15))

    fig.suptitle('Confidence Signals by Altitude', fontsize = 16)

    snru = data[get_vector_cols('snru')].values
    snrv = data[get_vector_cols('snrv')].values
    
    ax[0].plot(ALTITUDES, snru.mean(axis = 0), label = 'u')
    ax[0].plot(ALTITUDES, snrv.mean(axis = 0), label = 'v')
    ax[0].legend()
    ax[0].set_ylabel('Signal to Noise Ratio')
    ax[0].set_xlabel('Altitude (m)')
    
    sdu = data[get_vector_cols('sdu')].values
    sdv = data[get_vector_cols('sdv')].values
    
    ax[1].plot(ALTITUDES, sdu.mean(axis = 0), label = 'u')
    ax[1].plot(ALTITUDES, sdv.mean(axis = 0), label = 'v')
    ax[1].legend()
    ax[1].set_ylabel('Standard Deviation')
    ax[1].set_xlabel('Altitude (m)')

    r = data[get_vector_cols('r')].values
    ax[2].plot(ALTITUDES, r.mean(axis = 0))
    
    ax[2].set_ylabel('Reliability Factor')
    ax[2].set_xlabel('Altitude (m)')
    fig.savefig(Path(output_dir) / (prefix + '_' + name))

def wind_video(data, date:str = '04/21/2022', output_dir: str = ''):
    day = data[data['date'] == date].sort_values('time ending')

    t = day['time ending'].values
    u = day[get_vector_cols('u')].values
    v = day[get_vector_cols('v')].values
    w = day[get_vector_cols('w')].values

    conditions = day['macro_conditions'].values
    gust = day['macro_wg'].values

    x = np.zeros(u.shape[1])
    y = np.zeros(v.shape[1])
    z = ALTITUDES

    u0 = u[0, :]
    v0 = v[0, :]
    w0 = w[0, :]

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_xlim(-30.0, 30.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')

    global Q, text
    Q = ax.quiver(x, y, z, u0, v0, w0)

    text_x = -70.0
    text_y = 0.0
    text_z = 250.0
    text = ax.text(text_x, text_y, text_z, f'Time: {t[0]}' + f'\nCondition: {conditions[0]}' + f'\nGust: {gust[0]}')

    def update_quiver(i):
        print(i)
        global Q
        global text

        Q.remove()
        text.remove()

        ui = u[i, :]
        vi = v[i, :]
        wi = w[i, :]
        Q = ax.quiver(x, y, z, ui, vi, wi)
        text = ax.text(text_x, text_y, text_z, f'Time: {t[i]}' + f'\nCondition: {conditions[i]}' + f'\nGust: {gust[i]}')
        return

    ani = animation.FuncAnimation(fig, update_quiver, frames = range(len(t)), interval = 50)
    writergif = animation.PillowWriter(fps = 30)

    dashed_date = datetime.strptime(date, '%m/%d/%Y').date()
    ani.save(f'{dashed_date}.gif', writer = writergif)
    return None

def wind_component_velocity_violin(data: pd.DataFrame, output_dir: str = '', component: str = 'U', prefix: str = '', name: str = '_velocity_violin.png'):
    assert component.upper() in ['U', 'V']

    fig, ax = plt.subplots(1, 1, figsize = (15, 5))
    ax.set_title(f'{component.upper()} Component Velocity')
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Velocity (m/s)')
    violin = sns.violinplot(data[get_vector_cols(component.lower())].values, ax = ax, inner = 'point')
    violin.set_xticks(range(len(ALTITUDES)))
    violin.set_xticklabels(ALTITUDES)
    plt.xticks(rotation = 45, ha = 'right')
    fig.tight_layout()

    name = f'{component.lower()}_velocity_violin.png'
    fig.savefig(Path(output_dir) / (prefix + '_' + name))
    return None

def wind_bivariate_velocity_plot(data: pd.DataFrame, output_dir: str = '', prefix: str = '', name: str = 'wind_bivariate_velocity_plot.png'):
    u = data[get_vector_cols('u')].values.mean(1)
    v = data[get_vector_cols('v')].values.mean(1)

    g = sns.JointGrid(xlim = (-10.0, 20.0), ylim = (-20.0, 20.0))
    sns.scatterplot(x = u, y = v, color = 'black', s = 5, alpha = 0.4, ax = g.ax_joint)
    sns.kdeplot(x = u, y = v, color = 'red', fill = True, alpha = 0.4, ax = g.ax_joint)
    sns.kdeplot(x = u, fill = True, ax = g.ax_marg_x)
    sns.kdeplot(y = v, fill = True, ax = g.ax_marg_y)
    g.set_axis_labels('U Velocity (m/s)', 'V Velocity (m/s)')
    g.figure.tight_layout()
    g.figure.savefig(Path(output_dir) / (prefix + '_' + name))
    return None