import numpy as np
from scipy.spatial.distance import squareform, pdist
from utility import tracker

class simulator_model:
	def __init__(self, env_state, env_funs):
		self.env_state = env_state
		# store initial environment state upon initialization
		self.env_state0 = env_state.copy()
		self.env_state = env_state
		self.env_state0 = env_state.copy()
		self.N = env_state['N']
		self.locs = env_state['locs']
		self.bw = env_state['bw']
		self.nc = env_state['nc']
		self.T = env_state['T']
		self.beta_ = env_state['beta_']
		self.betahat = self.beta_[:,0]
		self.gamma_ = env_state['gamma_']
		self.alpha_ = env_state['alpha_']
		# Generate the flow matrix between populations
		self.Movers = env_state['Movers'][0]

		self.vac_fun = env_funs[0]
		self.test_fun = env_funs[1]
		# self.tracker = tracker()
		# self.tracker.update(self.env_state['S'],self.env_state['I'],self.env_state['R'],self.env_state['dI'])



	def __copy__(self):
		return simulator_model(self.env_state, [self.vac_fun, self.test_fun])

	def reset(self):
		self.env_state = self.env_state0

	def obs_fun(self,xtest):
		# true percent of population infected
		p = self.env_state['I']/self.N

		# probability of having symptoms given that you are positive
		a = self.env_state['a']
		# probability of having symptoms given that you are negative
		b = self.env_state['b']
		# base probability of getting tested given that you are symptomatic
		cc = self.env_state['cc']
		# base prob getting tested given that you are asymptomatic
		dd = self.env_state['dd']
		# probability of a false positive test
		fp = self.env_state['fp']
		# probability of a false negative test
		fn = self.env_state['fn']

		# probability of getting tested given that you are symptomatic
		c = cc + (xtest / self.env_state['N']) * (1 - cc)
		# probability of getting tested given that you are asymptomatic
		d = dd + (xtest / self.env_state['N']) * (1 - dd)

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

	def trans_function(self,xvac):
		# input the vaccination decision to the environment
		xi = self.env_state['xi']
		Movers = self.Movers
		if np.sum(Movers) == 0:
			Mhat = Movers
		else:
			Mhat = Movers / np.dot(Movers, np.ones(self.nc))

		beta = self.betahat
		gamma = self.gamma_

		S = self.env_state['S'].copy()
		I = self.env_state['I'].copy()
		R = self.env_state['R'].copy()
		N = self.env_state['N'].copy()

		# effective number of people vaccinated
		effective_vac = np.min([S.copy(), xi * xvac], axis=0)

		# post decision susceptible population
		Sx = S - effective_vac
		bhat = beta / N
		bvec = bhat * I

		# transition functions
		t1 = bvec * Sx
		t2 = self.alpha_ * np.dot(Mhat, bvec) * Sx
		# t2 = np.zeros(self.nc)
		S1 = Sx - t1 - t2
		I1 = (1-gamma) * I + t1 + t2
		R1 = R + gamma * I + effective_vac
		dI = I1 - I

		# update environmental state variables
		# self.tracker.update(S1, I1, R1, dI)
		self.env_state['S'] = S1.copy()
		self.env_state['I'] = I1.copy()
		self.env_state['R'] = R1.copy()
		self.env_state['dI'] = dI.copy()
		return np.sum(I1-I)

	def forward_one_step(self, xvac, t):
		self.betahat = self.env_state['beta_'][:,t]
		self.Movers = self.env_state['Movers'][0]
		# update nvac value
		self.env_state['nvac'] = self.vac_fun(t)
		self.env_state['ntest'] = self.test_fun(t)
		# transition using vaccine decisions
		return self.trans_function(xvac)

