import numpy as np
import data_loader as dl
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt

class DATA_obj:
	def __init__(self, list):
		self.data_list = list


class Simulator:
	def __init__(self, env_state,nc,T):
		# T = time horizon
		# nc = number of counties
		# beta = the transmission rate of the virus (expected number of interactions between infected-susc persons)
		# gamma = the recovery rate of the virus (1 / time of recovery)
		# alpha = the mobility rate of people traveling
		# n_seeds is the number of infected counties to start with
		# p_inf = percent of infected people to initialize per county
		# p_rec = percent of recovered people to initialize per county
		self.env_state = env_state
		self.env_state0 = env_state.copy()
		self.N = env_state['N']
		self.nc = nc
		self.T = T
		self.beta_ = env_state['beta_']
		self.gamma_ = env_state['gamma_']
		self.alpha_ = env_state['alpha_']
		self.p_inf = env_state['p_inf']
		self.p_rec = env_state['p_rec']
		self.p_susc = env_state['p_susc']
		#
		# self.FLOW = np.exp(-0.5 * pdist(self.locs / self.bw, 'sqeuclidean'))
		# self.FLOW = squareform(self.FLOW)
		# self.FLOW = np.min(
		# 	[0.8 * np.ones(self.FLOW.shape), np.max([0.001 * np.ones(self.FLOW.shape), self.FLOW], axis=0)],
		# 	axis=0)
		self.FLOW = dl.load_data('USA_DATA/FLOW_STATE.obj')
		# county populations

		# initialize SIR population to total populations for S and 0 IR
		self.S = np.zeros([nc, T])
		self.Sx = np.zeros([nc, T])
		self.I = np.zeros([nc, T])
		self.R = np.zeros([nc, T])
		self.V = np.zeros([nc, T])

		# initialize epidemic
		self.I[:, 0] = env_state['N'] * env_state['p_inf']
		self.R[:, 0] = env_state['N'] * env_state['p_rec']
		self.S[:, 0] = env_state['N'] * env_state['p_susc']
		self.marg_diff = env_state['N'] * env_state['p_inf']
		self.sss = 0.001
		self.Nvals = self.N.repeat(self.nc).reshape(self.nc, self.nc)


	def reset(self):
		self.env_state = self.env_state0
		self.S[:,1:] = 0
		self.I[:,1:] = 0
		self.R[:, 1:] = 0
		self.marg_diff = self.env_state['N'] * self.env_state['p_inf']

	def sample_Ihat(self,xtest):

		p = self.env_state['p_inf']

		# probability of having symptoms given that you are positive
		a = 0.9
		# probability of having symptoms given that you are negative
		b = 0.1
		cc = 0.7
		dd = 0.3
		# probability of a false positive test
		fp = 0
		# probability of a false negative test
		fn = 0

		# probability of getting tested given that you are symptomatic
		c = cc + (xtest / self.N) * (1 - cc)
		# probability of getting tested given that you are asymptomatic
		d = dd + (xtest / self.N) * (1 - dd)

		# probability of having symptoms
		v = a * p + b * (1 - p)
		# probability of getting tested
		pT = c * v + d * (1 - v)
		# probability of being positive and having symptoms
		n = (a * p) / v
		# probability of being negative and having symptoms
		m = (b * (1 - p)) / v

		# probability of wanting a test given that you are positive
		g = a * c + (1 - a) * d
		h = b * c + (1 - b) * d

		pa = g * p / pT

		# probability of testing positive given that you are tested
		pt = (1 - fn) * pa + (fp) * (1 - pa)
		# probability of testing negative given that you are tested
		nt = 1 - pt

		l2 = h - g
		l3 = (1 - fn - fp) * g * xtest - fp * xtest * (h - g)
		l0 = h
		l1 = -fp * xtest * h

		xtest = np.int32(xtest)
		Ihat = np.random.binomial(xtest, pt)
		return Ihat

	def trans_function(self,xvac,t):
		zeta = self.env_state['vac_effic']
		Mhat = np.random.binomial(np.int32(self.Nvals.T), self.FLOW)
		# Mhat2 = Mhat / self.Nvals.T
		a = np.random.uniform(low=-0.01, high=0.01,size=self.nc)
		b = np.random.uniform(low=-0.01, high=0.01, size=self.nc)
		# random beta
		beta = np.max([np.zeros(self.nc), np.min([np.ones(self.nc), self.env_state['beta_'] + a], axis=0)], axis=0)
		# random gamma
		gamma = np.max([np.zeros(self.nc), np.min([np.ones(self.nc), self.env_state['gamma_'] + b], axis=0)], axis=0)


		pS = self.env_state['p_susc']
		pI = self.env_state['p_inf']
		pR = self.env_state['p_rec']
		N = self.env_state['N']

		xvac_eff = np.min([N*pS, xvac],axis=0)

		Sx = N * pS - zeta * xvac_eff
		bhat = beta / N
		bvec = bhat * N * pI

		# transition functions
		t1 = bvec * Sx
		t2 = self.alpha_ * (Sx * np.dot(Mhat,bvec)) / (N + np.dot(Mhat,np.ones(self.nc)))
		# t2 = self.alpha_ * (Sx * np.dot(Mhat2,bvec))
		S1 = np.max([np.zeros(len(N)), Sx -  t1 - t2], axis=0)
		I1 = (1-gamma) * N * pI + t1 + t2
		R1 = N*pR + gamma * N*pI + zeta * xvac_eff
		self.marg_diff = t1+t2
		# update environmental state variables
		self.env_state['p_susc'] = S1 / N
		self.env_state['p_inf'] = I1 / N
		self.env_state['p_rec'] = R1 / N

		self.S[:, t + 1] = S1
		self.I[:, t + 1] = I1
		self.R[:, t + 1] = R1

		self.V[:,t+1] = xvac


		return self.S[:,t+1], self.I[:,t+1], self.R[:,t+1]


	def forward_one_step(self,xvac,nvac,t):
		# update nvac value
		self.env_state['nvac'] = nvac
		# transition using vaccine decisions
		self.trans_function(xvac,t)
		return np.sum(self.marg_diff)


	def plot_totals(self, NVac):
		plt.title('Epidemic Curves for ' + str(self.nc) + ' counties')
		plt.plot(np.sum(self.S, axis=0), 'b')
		plt.plot(np.sum(self.I, axis=0), 'r')
		plt.plot(np.sum(self.R, axis=0), 'g')
		plt.plot(np.sum(self.V, axis=0), 'k')
		plt.plot(NVac * np.ones(self.T), 'c--')
		plt.legend(["Susc", "Inf", "Rec", "Vac", "Vaccine Limit"])

	def getEnvState(self):
		return self.env_state


	def plot_net(self, ii=None):
		plt.plot(self.locs[:, 0], self.locs[:, 1], 'c*')
		if ii is not None:
			plt.plot(self.locs[ii, 0], self.locs[ii, 1], 'r*')
		for i in range(self.nc):
			for j in np.arange(i, self.nc):
				alpha = (self.FLOW[i, j] - np.min(self.FLOW)) / (np.max(self.FLOW) - np.min(self.FLOW))
				alpha = np.min([1, alpha + 0.01])
				plt.plot([self.locs[i, 0], self.locs[j, 0]], [self.locs[i, 1], self.locs[j, 1]], 'b', alpha=alpha)

	def plot_county_grid(self, I, J, title):
		fig, ax = plt.subplots(I, J)
		ax[0][0].set_xlabel(title)
		if I > 1:
			for i in np.arange(0, I):
				for j in np.arange(0, J):
					if ((J * i + j) < self.S.shape[0]):
						ax[i][j].plot(self.S[J * i + j, :], 'b')
						ax[i][j].plot(self.I[J * i + j, :], 'r')
						ax[i][j].plot(self.R[J * i + j, :], 'g')
						ax[i][j].plot(self.V[J * i + j, :], 'k')
						ax[i][j].set_title('beta = ' + str(self.beta_[J * i + j]))

