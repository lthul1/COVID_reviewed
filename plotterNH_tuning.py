import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt

time01 = '07,44,18'
time02 = '08,14,50'
time03 = '07,49,18'
time04 = '07,51,51'
time12 = '07,40,11'
time13 = '07,14,03'
time14 = '07,04,09'
time23 = '09,34,37'
time24 = '11,10,07'
time34 = '11,59,59'

def getDATA(timestr):
    null_time = '14,58,49'
    ncosts = dl.load_data('data/dataset_' + str(null_time) + '.obj')
    nparameter_list = dl.load_data('data/paramlist_' + str(null_time) + '.obj')

    nc = ncosts[0][0][0]
    print(nc)

    costs01 = dl.load_data('NH_folder/mcosts_NH_'+ timestr +'.obj')
    params01 = dl.load_data('NH_folder/params_NH_'+ timestr +'.obj')

    param0 = np.unique(params01[:,0])
    param1 = np.unique(params01[:,1])
    ids0 = {param0[i]:i for i in range(len(param0))}
    ids1 = {param1[i]:i for i in range(len(param1))}

    M = np.zeros([len(param0), len(param1)])
    counter=0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ci = ids0[params01[counter,:][0]]
            cj = ids1[params01[counter, :][1]]

            M[ci,cj] = costs01[counter]
            counter+=1
    param0 = np.round(param0,2)
    param1 = np.round(param1,2)
    Mc = (nc - M) / nc
    return param0, param1, Mc

def plotmap(param0, param1, M, axs,l0,l1, vmin=None, vmax=None):
    # extent = [np.min(param0), np.max(param0), np.min(param1), np.max(param1)]
    xticks = [str(param1[i]) for i in range(len(param1))]
    yticks = [str(param0[i]) for i in range(len(param0))]
    axs.imshow(M, origin='lower', cmap='turbo', vmin=vmin, vmax=vmax)
    axs.set_xticks(np.arange(len(param1)))
    axs.set_yticks(np.arange(len(param0)))
    axs.set_xticklabels(xticks)
    axs.set_yticklabels(yticks)

    axs.set_xlabel(l1)
    axs.set_ylabel(l0)

# param0 maps
param01, param11, M01 = getDATA(time01)
param02, param12, M02 = getDATA(time02)
param03, param13, M03 = getDATA(time03)
param04, param14, M04 = getDATA(time04)
fig,ax = plt.subplots(2,2)
plotmap(param01, param11[:11], M01[:,:11], ax[0][0], '$\\theta_0$', '$\\theta_1$')
plotmap(param02, param12[:11], M02[:,:11], ax[0][1], '$\\theta_0$', '$\\theta_2$')
plotmap(param03, param13[:11], M03[:,:11], ax[1][0], '$\\theta_0$', '$\\theta_3$', vmin=.42)
plotmap(param04, param14[:11], M04[:,:11], ax[1][1], '$\\theta_0$', '$\\theta_4$')

# param1 maps
param01, param11, M01 = getDATA(time12)
param02, param12, M02 = getDATA(time13)
param03, param13, M03 = getDATA(time14)
param04, param14, M04 = getDATA(time01)
fig2,ax2 = plt.subplots(2,2)
plotmap(param01, param11, M01, ax2[0][0], '$\\theta_1$', '$\\theta_2$')
plotmap(param02, param12[1:12], M02[:,1:12], ax2[0][1], '$\\theta_1$', '$\\theta_3$', vmin=.431)
plotmap(param03, param13[4:15], M03[:,4:15], ax2[1][0], '$\\theta_1$', '$\\theta_4$')

# # param2 maps
param01, param11, M01 = getDATA(time23)
param02, param12, M02 = getDATA(time24)
param03, param13, M03 = getDATA(time34)

fig3,ax3 = plt.subplots(2,2)
plotmap(param01, param11, M01, ax3[0][0], '$\\theta_2$', '$\\theta_3$')
plotmap(param02[:10], param12, M02[:10,:], ax3[0][1], '$\\theta_2$', '$\\theta_4$')
plotmap(param03[:10], param13[1:11], M03[:10, 1:11], ax3[1][0], '$\\theta_3$', '$\\theta_4$')
#
plt.show()

