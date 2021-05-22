import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt

time01 = '12,19,35'
time02 = '12,45,03'
time03 = '12,28,54'
time04 = '11,30,07'
time12 = '12,55,06'
time13 = '12,59,45'
time14 = '10,27,40'
time23 = '11,45,26'
time24 = '11,50,13'
time34 = '07,10,10'

def getDATA(timestr):
    null_time = '14,39,04'
    ncosts = dl.load_data('USA_OUTPUT/dataset_' + str(null_time) + '.obj')
    nparameter_list = dl.load_data('USA_OUTPUT/paramlist_' + str(null_time) + '.obj')

    nc = ncosts[0][0][0]
    print(nc)

    costs01 = dl.load_data('USA_risk/mcosts_USA_'+ timestr +'.obj')
    params01 = dl.load_data('USA_risk/params_USA_'+ timestr +'.obj')

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

def plotmap(param0, param1, M, axs,l0,l1):
    # extent = [np.min(param0), np.max(param0), np.min(param1), np.max(param1)]
    xticks = [str(param1[i]) for i in range(len(param1))]
    yticks = [str(param0[i]) for i in range(len(param0))]
    axs.imshow(M, origin='lower', cmap='turbo')
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
plotmap(param01, param11, M01, ax[0][0], '$\\theta_0$', '$\\theta_1$')
plotmap(param02, param12, M02, ax[0][1], '$\\theta_0$', '$\\theta_2$')
plotmap(param03, param13, M03, ax[1][0], '$\\theta_0$', '$\\theta_3$')
plotmap(param04, param14, M04, ax[1][1], '$\\theta_0$', '$\\theta_4$')

# param1 maps
param01, param11, M01 = getDATA(time12)
param02, param12, M02 = getDATA(time13)
param03, param13, M03 = getDATA(time14)
param04, param14, M04 = getDATA(time01)
fig2,ax2 = plt.subplots(2,2)
plotmap(param01, param11, M01, ax2[0][0], '$\\theta_1$', '$\\theta_2$')
plotmap(param02, param12, M02, ax2[0][1], '$\\theta_1$', '$\\theta_3$')
plotmap(param03, param13, M03, ax2[1][0], '$\\theta_1$', '$\\theta_4$')

# param2 maps
param01, param11, M01 = getDATA(time23)
param02, param12, M02 = getDATA(time24)
param03, param13, M03 = getDATA(time34)

fig3,ax3 = plt.subplots(2,2)
plotmap(param01[:10], param11[2:13], M01[:10,2:13], ax3[0][0], '$\\theta_2$', '$\\theta_3$')
plotmap(param02[:10], param12, M02[:10,:], ax3[0][1], '$\\theta_2$', '$\\theta_4$')
plotmap(param03, param13, M03, ax3[1][0], '$\\theta_3$', '$\\theta_4$')

Mtest = np.linspace(np.min(np.array([np.min(M01),np.min(M02), np.min(M03)])), np.max(np.array([np.max(M01),np.max(M02), np.max(M03)])), 10)
plt.figure()
plt.imshow(np.vstack([Mtest, Mtest]), 'jet')
plt.colorbar()




plt.show()

