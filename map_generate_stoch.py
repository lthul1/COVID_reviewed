from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]


ds= pd.read_csv('USA_DATA/fips_states.csv')
codes = ds.ss
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.offline as offline
import data_loader as dl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
T = 30
I = dl.load_data('USA_DATA/counties_51.obj')
I = np.array(I)
fips = dl.load_data('USA_DATA/state_fips.obj')
In =I[0,1,:,:]
# In[8,:] = np.max(I)
# In[39,:10] = np.min(I)
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
# set1 = In.copy()
# rmin = np.min(set1)
# rmax = np.max(set1)
# # ik = np.min([rmax*np.ones(set1.shape[0]), np.max([rmin * np.ones(set1.shape[0]), set1], axis=0)], axis=0)
# a = 0.9
# tmin = rmin
# tmax = (1 - a)*rmax
# ik = ((set1 - rmin) / (rmax - rmin)) * (tmax-tmin) + tmin
# ik = np.max([np.zeros(ik.shape), np.log10(ik)], axis=0)
# ik[8,:] = np.log10(np.max(I))
# # ik[39,:] = np.max([0, np.log10(np.min(I))])
In = np.max([np.zeros(In.shape), np.log10(In)], axis=0)
plotmap = True
if plotmap:
    data_slider = []
    # norm = matplotlib.colors.Normalize(vmin=np.min(In), vmax=np.max(In))
    # cmap = matplotlib.cm.get_cmap('GnBu')
    # median = np.median(In)
    # color = 'rgb' + str(cmap(norm(median))[0:3])
    colorbar=dict(tickvals = [3,4,5,6,7,8],
                      ticktext = ['1000', '10000', '50000', '100k', '500k','1M'])
    for t in range(T):
        # set1 = np.max([np.ones(In.shape[0]), np.log10(In[:, t])], axis=0)
        # set1 = ik[:, t]
        # rmin = np.min(set1)
        # rmax = np.max(set1)
        # # ik = np.min([rmax*np.ones(set1.shape[0]), np.max([rmin * np.ones(set1.shape[0]), set1], axis=0)], axis=0)
        # a = 0.9
        # tmin = (1 + (1-a))*rmin
        # tmax = (1 - a)*rmax
        # ik = ((set1 - rmin) / (rmax - rmin)) * (tmax-tmin) + tmin
        # ik = np.log10(ik)
        data_each_yr = dict(
            type='choropleth',
            locations=codes,
            z=In[:,t],
            colorbar=colorbar,
            colorscale="YlOrRd",
            locationmode='USA-states',
            zmin=3,
            zmax=1.3*np.log10(np.max(I))
            # autocolorscale=True
        )

        data_slider.append(data_each_yr)

    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='Day {}'.format(i))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

    layout = dict(title ='Virus cases', geo=dict(scope='usa',
                           projection={'type': 'albers usa'}),
                  sliders=sliders)

    fig = dict(data=data_slider, layout=layout)
    # fig = dict(data=data_slider)
    offline.plot(fig)
#
# fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
#                            color_continuous_scale="Viridis",
#                            range_color=(0, 12),
#                            scope="usa",
#                            labels={'unemp':'unemployment rate'}
#                           )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()