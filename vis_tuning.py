import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl


param0 = np.arange(0.05, 0.95, .1)
param1 = np.arange(2,10,1)
print(param0)
print(param1)

parameterlist = np.vstack(np.meshgrid(param0,param1)).reshape(2,len(param0)*len(param1)).T

time = '15,07,37'
name = 'nonlinear'
pdata = dl.load_data('tuning_data/params_'+str(name)+'_'+time+'.obj')
costs = dl.load_data('tuning_data/costs_'+str(name)+'_'+time+'.obj')
mcosts = dl.load_data('tuning_data/mcosts_'+str(name)+'_'+time+'.obj')
scosts = dl.load_data('tuning_data/scosts_'+str(name)+'_'+time+'.obj')

ids0 = {param0[i]:i for i in range(len(param0))}
ids1 = {param1[i]:i for i in range(len(param1))}

M = np.zeros([len(param0), len(param1)])
M1 = np.zeros([len(param0), len(param1)])
M2 = np.zeros([len(param0), len(param1)])

counter = 0
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ci = ids0[pdata[counter,:][0]]
        cj = ids1[pdata[counter, :][1]]

        M[ci,cj] = costs[counter]
        M1[ci, cj] = mcosts[counter]
        M2[ci, cj] = scosts[counter]
        counter+=1

pflag = True
if pflag:
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(M, origin='lower')
    ax[1].imshow(M1, origin='lower')
    ax[2].imshow(M2, origin='lower')
# plt.plot(M.flatten())
else:
    ax = plt.axes(projection='3d')
    X,Y = np.meshgrid(param0, param1)
    ax.plot_surface(X,Y,M1)

fig2,ax2 = plt.subplots(1,1)
ax2.plot(param0, M[:,0])


plt.show()


