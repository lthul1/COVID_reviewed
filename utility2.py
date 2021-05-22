import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import data_loader as dl
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.offline as offline
import data_loader as dl
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

def gen_betas(nc,T,n, seed):
	# this function takes number of counties as input and returns the mean beta values for each county
	# stock betas
	deltabeta = 0.05
	np.random.seed(seed)
	betaz = np.random.uniform(low=0.2, high=0.4, size = nc)
	np.random.seed(None)
	betahat = np.zeros([nc,T,n])
	betahat[:,0,0] = betaz
	for i in range(n):
		for t in range(T-1):
			a = np.random.uniform(low=-deltabeta, high=deltabeta, size=nc)
			betahat[:,t+1,i] = betahat[:,t,i] + a

	betahat = np.max([0.3 * np.ones([nc,T,n]), np.min([0.8 * np.ones([nc,T,n]), betahat], axis=0)], axis=0)
	return betahat

def gen_betas_wrong(nc,T,n, seed):
	# this function takes number of counties as input and returns the mean beta values for each county
	# stock betas
	deltabeta = 0.05
	np.random.seed(seed)
	betaz = np.random.uniform(low=0.2, high=0.25, size = nc)
	np.random.seed(None)
	betahat = np.zeros([nc,T,n])
	betahat[:,0,0] = betaz
	for i in range(n):
		for t in range(T-1):
			a = np.random.uniform(low=-deltabeta, high=deltabeta, size=nc)
			betahat[:,t+1,i] = betaz

	betahat = np.max([0.3 * np.ones([nc,T,n]), np.min([0.8 * np.ones([nc,T,n]), betahat], axis=0)], axis=0)
	return betahat

def gen_USA_betas(nc,T,n):
	# this function takes number of counties as input and returns the mean beta values for each county
	# stock betas
	deltabeta = 0.04
	betaz = dl.load_data('USA_DATA/state_beta.obj')
	betahat = np.zeros([nc, T, n])
	betahat[:, 0, 0] = betaz
	for i in range(n):
		for t in range(T - 1):
			a = np.random.uniform(low=-deltabeta, high=deltabeta, size=nc)
			betahat[:, t + 1, i] = betaz + a

	betahat = np.min([0.95 * np.ones([nc, T, n]), np.max([0.5 * np.ones([nc, T, n]), betahat], axis=0)], axis=0)

	return betahat



def gen_FLOW(nc,T,n,FLOW,N):
	# this function takes number of counties as input and returns the mean beta values for each county
	# stock betas
	Nvals = N.repeat(nc).reshape(nc, nc)
	Movers = [np.random.binomial(np.int32(Nvals.T), FLOW) for _ in range(T)]
	return Movers

def gen_N(nc):
	# this function takes number of counties as input and returns the mean beta values for each county
	# stock betas
	np.random.seed(0)
	maxpop = 2500
	N = np.array([700, 800, 845, 1239, 845,1300,2000])
	if nc < len(N):
		np.random.shuffle(N)
		N = N[:nc]
	else:
		N = np.random.randint(low = 200, high=maxpop,size=nc)
	return N

def gen_locs(nc):
	loc = np.array([[0.2, 0.42, 0.6, 0.92, 0.55], [0.82, 0.33, 0.1, 0.12, 0.88]]).T
	if nc<loc.shape[0]:
		locs = loc[:nc,:]
	else:
		np.random.seed(5)
		locs = np.random.rand(nc, 2)
	return locs


def NVac_const(nc, t, a):
	return nc * 100

def NTest_const(nc, t, a):
	return 100

def NVac_t(t, T):
	a = np.zeros(T)
	tt = np.arange(0,T)
	start = 2
	a[start:] = 25*tt[start:] + 40
	return a[t]

def stochNVac_t(nc, t, T):
	c=10
	np.random.seed(c)
	a = np.random.normal(50, 10, size=T)
	return np.int32(10*nc*t + nc*a[t])

class vaccine_process:
	def __init__(self, nc, T):
		c = 10
		np.random.seed(c)
		self.a = np.random.normal(20, 10, size=T)
		self.T = T
		self.nc = nc
		np.random.seed(None)

	def stoch(self, t):
		return np.int32(self.nc * t + self.nc * self.a[t]/10)

	def stoch2(self, t):
		return np.int32(3 * self.nc * t + self.nc * self.a[t]/50)

	def const(self, t):
		return self.nc * 25

	def stoch_USA(self, t):
		return np.int32(1000 * self.nc * t + self.nc * self.a[t])




class vaccine_process2:
	def __init__(self, nc, T, N):
		self.NT = np.sum(N)
		self.pct = 0.03
		c = 10
		np.random.seed(c)
		self.a = np.random.normal(25, 5, size=T)
		self.T = T
		self.nc = nc
		np.random.seed(None)

		self.nvacs = dl.load_data('USA_DATA/nvac_list.obj')


	def stoch(self, t):
		v = np.int32(self.pct * self.NT + self.a[t] * t)
		return v

	def stoch2(self, t):
		return np.int32(3 * self.nc * t + self.nc * self.a[t]/50)

	def const(self, t):
		return self.nc * 50

	def stoch_USA(self, t):
		return np.int32(1000 * self.nc * t + self.nc * self.a[t])

	def data_vacs(self,t):
		return self.nvacs[t]


class test_process:
	def __init__(self, nc, T, N):
		self.T = T
		self.nc = nc
		c = 10
		self.NT = np.sum(N)
		self.pct = .1
		np.random.seed(c)
		self.a = np.random.normal(50, 50, size=T)
		np.random.seed(None)
		self.ntest = dl.load_data('USA_DATA/ntest_list.obj')

	def const(self, t):
		return np.int32(self.pct * self.NT)

	def const_USA(self, t):
		return self.nc * 1000

	def stoch_USA(self, t):
		return np.int32(50 * self.nc * t + self.nc * self.a[t])

	def data_tests(self, t):
		return self.ntest[t]


def simplex_projector(y, const):
	# y not in the simplex
	# const < 0
	nc = y.shape[0]
	pmodel1 = gp.Model("m1")
	x1 = pmodel1.addMVar(shape=nc, name="x")

	M = np.eye(nc)
	pmodel1.setParam('OutputFlag', 0)
	pmodel1.setObjective(x1 @ M @ x1 - 2 * y @ x1, GRB.MINIMIZE)
	pmodel1.addConstr(np.ones(nc) @ x1 == 1 - const, name="c")
	pmodel1.addConstr(x1 >= 0)
	pmodel1.optimize()
	return x1.X

def simplex_projector_integer(y, const):
	# y not in the simplex
	# const < 0
	nc = y.shape[0]
	pmodel1 = gp.Model("m1")
	x1 = pmodel1.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

	M = np.eye(nc)
	pmodel1.setParam('OutputFlag', 0)
	pmodel1.setObjective(x1 @ M @ x1 - 2 * y @ x1, GRB.MINIMIZE)
	pmodel1.addConstr(np.ones(nc) @ x1 == const, name="c")
	pmodel1.addConstr(x1 >= 0)
	pmodel1.optimize()
	return x1.X


def gradproj2(y, const):
	# y not in the simplex
	# const < 0

	if (y < 0).any():
		bools = y < 0
		y[bools] = 0

	if np.sum(y) > const:

		nc = y.shape[0]
		pmodel1 = gp.Model("m1")
		x1 = pmodel1.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

		M = np.eye(nc)
		pmodel1.setParam('OutputFlag', 0)
		pmodel1.setObjective(x1 @ M @ x1 - 2 * y @ x1, GRB.MINIMIZE)
		pmodel1.addConstr(np.ones(nc) @ x1 == const, name="c")
		pmodel1.addConstr(x1 >= 0)
		pmodel1.optimize()
		y = x1.X
	return y

def gradproj(y, const):
	# y not in the simplex
	# const < 0
	nc = y.shape[0]
	pmodel1 = gp.Model("m1")
	x1 = pmodel1.addMVar(shape=nc, vtype=GRB.CONTINUOUS, name="x")

	M = np.eye(nc)
	pmodel1.setParam('OutputFlag', 0)
	pmodel1.setObjective(x1 @ M @ x1 - 2 * y @ x1, GRB.MINIMIZE)
	pmodel1.addConstr(np.ones(nc) @ x1 == const, name="c")
	pmodel1.addConstr(x1 >= 0)
	pmodel1.optimize()
	y = x1.X
	return y

def proj(y, const):
	y /= const
	a = np.ones(y.shape)
	l = y/a
	idx = np.argsort(l)
	d = len(l)
	evalpL = lambda k: np.sum(a[idx[k:]]*(y[idx[k:]] - l[idx[k]]*a[idx[k:]]))-1

	def bisectsearch():
		idxL,idxH = 0, d-1
		L = evalpL(idxL)
		H = evalpL(idxH)

		if L<0:
			return idxL

		while (idxH - idxL)>1:
			iMid = int((idxL + idxH)/2)
			M = evalpL(iMid)

			if M >0:
				idxL,L = iMid, M
			else:
				idxH, H = iMid, M

		return idxH


	k = bisectsearch()
	lam = (np.sum(a[idx[k:]]*y[idx[k:]]) - 1)/np.sum(a[idx[k:]])

	x = np.maximum(0,y-lam*a)
	return x*const





class tracker:
	def __init__(self):
		self.S = []
		self.I = []
		self.R = []
		self.dI = []

	def update(self,S,I,R,dI):
		self.S.append(S)
		self.I.append(I)
		self.R.append(R)
		self.dI.append(dI)

	def getI0(self, I0):
		self.I0 = I0

	def plot(self, z):
		Ss = np.array(self.S)
		Is = np.array(self.I)
		Rs = np.array(self.R)
		dIs = np.array(self.dI)
		fig,ax = plt.subplots(1,3)
		ax[0].plot(Ss[:,z])
		ax[0].plot(Is[:,z])
		ax[0].plot(Rs[:,z])
		ax[0].plot(np.zeros(Rs.shape[0]), 'k')

		dItest = np.diff(Is[:,z])
		ax[1].plot(dIs[:,z])
		ax[1].plot(dItest)

		ax[2].plot(np.cumsum(dItest) + self.I0)
		ax[2].plot(Is[:,z])
		ax[2].plot(np.cumsum(dIs[:,z]))


def USA_plot(costs, T, scheme='inf'):
	colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
				  "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
				  "#08519c", "#0b4083", "#08306b"
				  ]

	ds = pd.read_csv('USA_DATA/fips_states.csv')
	codes = ds.ss

	fips = dl.load_data('USA_DATA/state_fips.obj')
	# In[8,:] = np.max(I)
	# In[39,:10] = np.min(I)
	scl = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]
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
	def sigmoid(x):
		return 1 / (1 + np.exp(-0.1* x))

	if scheme == 'vac':
		In = np.max([np.zeros(costs.shape), np.log10(costs+10e-6)], axis=0)
		In = In.T
	else:

		I = costs.T
		bool = I < 0
		I[bool] = -np.max([np.zeros(I[bool].shape), np.log10(-I[bool]+10e-6)], axis=0)
		I[~bool] = np.max([np.zeros(I[~bool].shape), np.log10(I[~bool] + 10e-6)], axis=0)
		In = -np.tanh(I/10)
	plotmap = True
	if plotmap:
		data_slider = []
		# norm = matplotlib.colors.Normalize(vmin=np.min(In), vmax=np.max(In))
		# cmap = matplotlib.cm.get_cmap('GnBu')
		# median = np.median(In)
		# color = 'rgb' + str(cmap(norm(median))[0:3])
		if scheme == 'vac':
			# In = sigmoid(In)
			vector = np.array([1, 2, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7])
			colorbar = dict(tickvals=list(vector),
							ticktext=['1', '100', '10k', '175k', '315k', '565k', '1M', '1.25M', '1.75M', '3.15M', '5.65M', '10M'])

			cmap = 'RdPu'

			zmin = 3.5
			zmax = np.log10(np.max(costs)+10e-6)
			# zmax = 1
		elif scheme == 'inf':
			colorbar = dict(tickvals=[0, 2, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7],
							ticktext=['0', '100', '10k', '175k', '315k', '565k', '1M', '1.25M', '1.75M', '3.15M',
									  '5.65M', '10M'])
			cmap = 'YlOrRd'
			zmin = 4.5
			zmax = np.log10(np.max(costs) + 10e-6)
		else:
			vectorl = np.array([-10e6, -10e5, -10e4, -10e3, -10e2, -10, 1, 10, 10e2, 10e3, 10e4, 10e5, 10e6])
			vector = vectorl.copy()
			bool2 = vector < 0
			vector[bool2] = -np.max([np.zeros(vector[bool2].shape), np.log10(-vector[bool2] + 10e-6)], axis=0)
			vector[~bool2] = np.max([np.zeros(vector[~bool2].shape), np.log10(vector[~bool2] + 10e-6)], axis=0)
			vector = -np.tanh(vector / 10)
			colorbar = dict(tickvals=list(vector),
							ticktext=['-1M', '-100k', '-10k', '-1k', '-100', '-10', '+1', '+10', '+100', '+1k', '+10k', '+100k', '1M'])
			cmap = 'RdYlBu'
			zmin = 1.5*np.min(In)
			zmax = np.max(In)

		for t in range(T-1):
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
				colorscale=cmap,
				locationmode='USA-states',
				zmin=zmin,
				zmax=zmax
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

		layout = dict(title='Virus cases', geo=dict(scope='usa',
													projection={'type': 'albers usa'}),
					  sliders=sliders)

		fig2 = dict(data=data_slider, layout=layout)
		# fig = dict(data=data_slider)
		offline.plot(fig2)