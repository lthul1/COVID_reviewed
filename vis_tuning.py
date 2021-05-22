import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl


param0 = np.arange(1, 4, .25)
param1 = np.arange(0, 2, .2)
print(param0)
print(param1)

parameterlist = np.vstack(np.meshgrid(param0,param1)).reshape(2,len(param0)*len(param1)).T

time = '15,42,45'
name = 'risk'
pdata = dl.load_data('tuning_data/params_'+str(name)+'_'+time+'.obj')
costs = dl.load_data('tuning_data/costs_'+str(name)+'_'+time+'.obj')
mcosts = dl.load_data('tuning_data/mcosts_'+str(name)+'_'+time+'.obj')
scosts = dl.load_data('tuning_data/scosts_'+str(name)+'_'+time+'.obj')
ids0 = {param0[i]:i for i in range(len(param0))}
ids1 = {param1[i]:i for i in range(len(param1))}

M = np.zeros([len(param0), len(param1)])
M1 = np.zeros([len(param0), len(param1)])
M2 = np.zeros([len(param0), len(param1)])
pdata = parameterlist

counter = 0

print(mcosts)
print(pdata.shape[1])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ci = ids0[pdata[counter,:][0]]
        cj = ids1[pdata[counter, :][1]]

        # M[ci,cj] = costs[counter]
        M1[ci, cj] = mcosts[counter]
        # M2[ci, cj] = scosts[counter]
        counter+=1
pflag = 0
if pflag == 0:
    fig,ax = plt.subplots(1,1)
    # extent = [np.min(param0), np.max(param0), np.min(param1), np.max(param1)]
    xticks = [str(round(param1[i],1)) for i in range(len(param1))]
    yticks = [str(round(param0[i],1)) for i in range(len(param0))]
    im = ax.imshow(M1, origin='lower')
    ax.set_xticks(np.arange(len(param1)))
    ax.set_yticks(np.arange(len(param0)))
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    fig.colorbar(im)
    ax.set_xlabel('theta_DLA_4')
    ax.set_ylabel('theta_DLA_3')
    # ax.plot(1,4, 'r*')
# plt.plot(M.flatten())
elif pflag == 1:
    ax = plt.axes(projection='3d')
    # t0 = np.arange(len(param0))
    # t1 = np.arange(len(param1))
    X,Y = np.meshgrid(param0,param1)
    ax.plot_surface(X,Y,M1.T)
    ax.set_xlabel('param0')
    ax.set_ylabel('param1')
    ax.set_xticks(param0)
    ax.set_yticks(param1)

else:
    plt.plot(param1, Fl)
fig2,ax2 = plt.subplots(1,1)
# ax2.plot(Fl)

ax2.plot(param0, M1[:,0])
ax2.plot(param0, M1[:,1])
ax2.plot(param0, M1[:,2])
plt.show()
