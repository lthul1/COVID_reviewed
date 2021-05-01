import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
# from utility import gradproj2 as gradproj

from utility import proj as gradproj


def simulator(statet, x, T):
    # x comes in as a T*nc vector

    nc = statet['nc']
    state = statet.copy()
    xl = [x[(nc * i):(nc * (i + 1))] for i in range(T)]
    beta = state['beta']

    gamma = 0.3

    cost = []
    for t in range(T):
        pS = state['pS'].copy()
        pI = state['pI'].copy()
        pR = state['pR'].copy()
        N = state['N']
        Sx = np.max([np.zeros(nc), N*pS - xl[t]], axis=0)
        S = Sx - (beta) * pI * Sx
        I = (1-gamma) * N * pI + (beta) * pI * Sx
        R = N*pR + gamma * N * pI + np.min([N*pS, xl[t]], axis=0)
        state['pS'] = S / N
        state['pI'] = I / N
        state['pR'] = R / N
        cost.append(np.sum(I - N*pI))
    cumuls = np.cumsum(cost)
    return np.cumsum(cumuls)[T-1]



nc = 5
maxpop=2500

np.random.seed(17)
N = np.random.randint(low = 200, high=maxpop,size=nc)

beta = np.random.uniform(0.6, 0.95, size = nc)
pI0 = 0.2*np.random.rand(nc)
pS0 = 1 - pI0
pR0 = np.zeros(nc)
T = 8

nv = 300
state = {'pS':pS0, 'pI':pI0, 'pR':pR0, 'N':N, 'nc':nc, 'nv':nv, 'beta':beta}
# x = [np.random.rand(nc) for _ in range(T)]
# x = [x/np.sum(x) for _ in range(T)][0]
np.random.seed(None)
x = np.random.rand(nc * T)
print(simulator(state,x,T))


K =1000

kk = np.arange(1,K+1)
gamma = 0.2
c = 0.1
# eta = np.max([0.01 * np.ones(eta.shape), hh/(eta+hh)], axis=0)
ck = c/(kk)**gamma
xtune1 = x.copy()
xtune_t = xtune1.copy()
mt = 0
vt = 0
b1 = 0.9
b2 = 0.99
eps= 10e-8
g2=0
lr = 8

et = 0.8
tlist = [list(xtune1)]

print('tunable1 = ' + str(xtune1))
F = []
for k in range(K):
    np.random.seed(None)
    if k%25==0:
        print('###  k = ' + str(k) + '  ####')

    vk = np.random.rand(xtune1.shape[0])
    hk = 2 * np.int32(vk > 0.5) - 1

    xtune_t =xtune1.copy()
    tune_tn1 = xtune_t - ck[k] * hk
    tune_t1 = xtune_t + ck[k] * hk
    tune_tn1 = np.array([gradproj(tune_tn1[(nc * i):(nc * (i + 1))], nv) for i in range(T)]).flatten()
    tune_t1 = np.array([gradproj(tune_t1[(nc * i):(nc * (i + 1))], nv) for i in range(T)]).flatten()

    F1 = simulator(state, tune_t1, T)
    Fn1 = simulator(state, tune_tn1, T)

    dG = (F1 - Fn1)/(2*ck[k]) * hk
    if k % 25 == 0:
        print('gradient = ' + str(dG[:nc]))
    grad_squared = dG * dG
    mt = b1 * mt + (1-b1)*dG
    vt = b2 * vt + (1-b2)*grad_squared
    mhat = mt/(1-b1**(k+1))
    vhat = vt/(1-b2**(k+1))
    alpha = lr / (np.sqrt(vhat) + eps)
    alphaG = alpha * mhat
    xtune1 = xtune_t - alphaG
    if k % 25 == 0:
        print()
        # print('pre projection tunable = ' +str(xtune1[:nc]))
    xtune1 = np.array([gradproj(xtune1[(nc * i):(nc * (i + 1))], nv) for i in range(T)]).flatten()
    if k % 25 == 0:
        print('tunable1 = ' + str(xtune1[:nc]))
        print('tunable1 = ' + str(xtune1[nc:2*nc]))
        # print('alpha = ' + str(alpha))
        print('ck = ' + str(ck[k]))
    tlist.append(list(xtune1))

    F.append(F1)

print([np.sum(np.array(tlist)[K-1,nc*(i):nc*(i+1)]) for i in range(T)])

fig,ax = plt.subplots(1,4)
ax[0].plot(np.array(tlist)[:,0])
ax[0].plot(np.array(tlist)[:,1])
ax[0].plot(np.array(tlist)[:,2])
ax[1].plot(np.array(tlist)[:,3])
ax[1].plot(np.array(tlist)[:,4])
ax[1].plot(np.array(tlist)[:,5])
ax[2].plot(np.array(tlist)[:,6])
ax[2].plot(np.array(tlist)[:,7])
ax[2].plot(np.array(tlist)[:,8])

xeven = N / np.sum(N) * nv
xe = np.array([xeven for i in range(T)]).flatten()
Feven = simulator(state, xe, T)


plt.plot(F)
plt.plot(Feven*np.ones(len(F)))
plt.show()














