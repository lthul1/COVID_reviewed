import numpy as np
import scipy.stats as stats
import gurobipy as gp
from gurobipy import GRB
from utility import *

class controller:
	def __init__(self, state, env_funs):
		# initialize controller state
		# bundle of environment functions
		# env_funs[0] is the observation function
		self.state = state
		self.state0 = state.copy()
		self.N = self.state['N']
		self.nc = self.state['nc']
		self.beta = self.state['beta']
		self.gamma = self.state['gamma']
		self.obs_fun = env_funs[0]
		self.env_funs = env_funs
		self.lz = self.state['lz']

	def __copy__(self):
		return controller(self.state.copy(), self.env_funs)

	def exogenous(self, xtest, t):
		# receive the new exogenous information
		return self.obs_fun(xtest)

	def transition(self, xvac, xtest, Ihat):
		# update the transition function after the implementation decision has been made
		eps = 10e-6
		N = self.N
		pS = self.state['pS']
		pI = self.state['pI']
		pR = self.state['pR']

		# compute the necessary satistics
		Sbar = pS * self.N
		Ibar = pI * self.N
		Rbar = pR * self.N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		s = np.sqrt(VarSI + VarxI + cov + eps)

		mean = EISx

		# E[I S^x]
		term = (self.beta / N) * (mean * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))


		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N*pS*(1-pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac)/sigma_susc) + sigma_susc*stats.norm.pdf((Sbar - xvac)/sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1-self.gamma) * Ibar + term
		Rbar1 = Rbar + self.gamma*Ibar + Emin

		# create the priors
		alpha_tz = self.state['lz'] * Ibar1
		beta_tz = self.state['lz'] * (N-Ibar1)

		# update the estimator for the beta-binomial
		p_inf = (Ihat + alpha_tz) / (xtest + self.state['lz'] * N)
		if (p_inf < 0).any():
			print('ISNN')

		# project the other dimensions back onto it
		sv = np.array([simplex_projector(np.array([Sbar1[i]/N[i], Rbar1[i]/N[i]]), const=p_inf[i]) for i in range(self.nc)])
		self.state['pS'] = sv[:,0]
		self.state['pR'] = sv[:,1]
		self.state['pI'] = p_inf



	def forward_one_step(self, xvac, xtest, nvac, ntest):
		# jump from time t to t+1
		self.state['t'] += 1
		Ihat = self.exogenous(xtest, self.state['t'])
		self.state['nvac'] = nvac
		self.state['ntest'] = ntest
		self.transition(xvac, xtest, Ihat)