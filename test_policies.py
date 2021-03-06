import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.stats as stats
from utility import simplex_projector, simplex_projector_integer
import matplotlib.pyplot as plt


class null_policy:
	# The null test policy assumes there is an infinite amount of tests
	def __init__(self, model, params, vacpol):
		# params is empty
		# model gives the base model
		self.model = model
		self.params = params
		self.vacpol = vacpol

	def update(self, model, params, vacpol):
		# params is empty
		# model updates the base model
		self.model = model
		self.vacpol = vacpol

	def decision(self, xvac):
		# xvac was the most recent vaccine decision (not needed for null)
		ntest = self.model.state['ntest']
		even = ntest / self.model.nc
		while np.sum(even) > ntest:
			j = np.random.randint(high = self.model.nc)
			even[j] -= 1
		return np.int32(even)

class prop_policy:
	# The null test policy assumes there is an infinite amount of tests
	def __init__(self, model, params, vacpol):
		# params is empty
		# model gives the base model
		self.model = model
		self.params = params
		self.vacpol = vacpol

	def update(self, model, params, vacpol):
		# params is empty
		# model updates the base model
		self.model = model
		self.vacpol = vacpol

	def decision(self, xvac):
		# xvac was the most recent vaccine decision (not needed for null)
		ntest = self.model.state['ntest']
		N = self.model.state['N']
		even = ntest * (N / np.sum(N))
		while np.sum(even) > ntest:
			j = np.random.randint(high = self.model.nc)
			even[j] -= 1
		return np.int32(even)


class thompson_sampling:
	def __init__(self, model, params, vacpol):
		# params is empty
		# model gives the base model
		self.model = model
		self.xsamps = params[0]
		self.vacpol = vacpol

	def update(self, model, params, vacpol):
		# params is empty
		# model updates the base model
		self.model = model

	def decision(self, xvac):
		# xvac was the most recent vaccine decision (not needed for null)
		cs = []
		vacpol = self.vacpol.__copy__()
		Xtest = np.random.rand(self.model.nc, self.xsamps)
		Xtest = Xtest / np.sum(Xtest,axis=0)
		for k in range(self.xsamps):
			xtest = Xtest[:,k]

			ntest = self.model.state['ntest']
			state1 = self.model.state.copy()

			N = self.model.state['N']
			xtest = np.int32(N * xtest)
			pS = self.model.state['pS']
			pI = self.model.state['pI']
			pR = self.model.state['pR']
			beta = self.model.state['beta']
			gamma = self.model.state['gamma']
			eps = 1e-6

			# compute the necessary satistics
			Sbar = pS * N
			Ibar = pI * N
			Rbar = pR * N

			EISx = N * pI * ((N - 1) * pS - xvac)
			ESI = N * (N - 1) * pI * pS
			ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
			VarSI = ES2I2 - ESI ** 2
			VarxI = xvac ** 2 * (N * pI * (1 - pI))
			cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
			s = np.sqrt(VarSI + VarxI + cov + eps)
			mean = EISx

			# E[I S^x]
			term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

			# E[max(0,S-x)] = E[S] - E[min(S,x)]
			sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
			ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
				(Sbar - xvac) / sigma_susc)
			Emin = ESx - Sbar

			# Get the predicted states at t+1
			Sbar1 = ESx - term
			Ibar1 = (1 - gamma) * Ibar + term
			Rbar1 = Rbar + gamma * Ibar + Emin

			nc = len(self.model.N)
			lz = self.model.state['lz']

			pbarI1 = np.min([(1 - 10e-6) * np.ones(nc), Ibar1/N + 10e-6], axis=0)

			Ihat = np.array([np.random.binomial(xtest[i], pbarI1[i]) for i in range(nc)])

			pI1 = (Ihat + lz * Ibar1) / (xtest + lz * N)

			sv = np.array([simplex_projector(np.array([Sbar1[i] / N[i], Rbar1[i] / N[i]]), const=pI1[i]) for i in
			               range(nc)])

			pS1 = sv[:, 0]
			pR1 = sv[:, 1]

			state1['pS'] = pS1
			state1['pI'] = pI1
			state1['pR'] = pR1

			model_temp = self.model.__copy__()
			model_temp.state = state1.copy()
			vacpol.update(model_temp, 0)
			c = np.sum(-gamma * N * pI1 + beta * pI1 * np.max([np.zeros(nc), (N-1)*pS1 - vacpol.decision()]))
			cs.append(c)
		kstar = np.argmin(cs)
		return Xtest[:,kstar]


class EI:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.params = params
		self.vacpol = vacpol

	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		state1 = self.model.state.copy()

		N = self.model.state['N']
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		nc = self.model.state['nc']
		beta = self.model.state['beta']
		gamma = self.model.state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		s = np.sqrt(VarSI + VarxI + cov + eps)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		nc = len(self.model.N)
		lz = self.model.state['lz']
		m = gp.Model("qp")
		x = m.addMVar(nc, vtype=GRB.INTEGER, name="x")
		pbarI1 = Ibar1/N
		Az = np.diag((N**2 * pbarI1 * (1-pbarI1)) /(lz * N + 1))
		c = (lz*N**3 * pbarI1 * (1-pbarI1)) /(lz * N + 1)

		m.setParam('OutputFlag', 0)
		m.setObjective(x @ Az @ x + c @ x, GRB.MAXIMIZE)

		M = np.ones(nc)
		Ntest = np.array([np.int32(ntest)])

		# Add constraints
		m.addConstr(M @ x == Ntest, name="c")
		m.addConstr(0 <= x)
		m.setParam('NonConvex', 2)
		m.optimize()
		xx = x.X
		return xx



class prop_greedy_trade:

	def __init__(self, model, params, vacpol):
		self.model = model
		self.pr = params[0]
		self.vacpol = vacpol

	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		pcts = self.model.N / sum(self.model.N)
		xtest = self.pr * ntest * pcts

		ntest1 = np.floor(ntest - np.sum(xtest))

		state1 = self.model.state.copy()

		N = self.model.state['N']
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		nc = self.model.state['nc']
		beta = self.model.state['beta']
		gamma = self.model.state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		s = np.sqrt(VarSI + VarxI + cov + eps)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		nc = len(self.model.N)
		lz = self.model.state['lz']
		m = gp.Model("qp")
		x = m.addMVar(nc, vtype=GRB.INTEGER, name="x")
		pbarI1 = Ibar1 / N
		Az = np.diag(pbarI1 * (1 - pbarI1))
		c = np.dot(Az, N)
		m.setParam('OutputFlag', 0)
		m.setObjective(x @ Az @ x + c @ x, GRB.MAXIMIZE)

		M = np.ones(nc)
		Ntest = np.int32(np.array([ntest]))

		# Add constraints
		m.addConstr(M @ x == Ntest, name="c")
		m.addConstr(0 <= x)
		m.setParam('NonConvex', 2)
		m.optimize()
		xx = x.X

		xtest = xtest + xx
		# print(xtest)
		return xtest

class REMBO_EI:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.proj_dim = np.int32(params[0])
		self.vacpol = vacpol
		self.A = np.random.rand(np.int32(self.model.nc), self.proj_dim)


	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		state1 = self.model.state.copy()

		N = self.model.state['N']
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		nc = self.model.state['nc']
		beta = self.model.state['beta']
		gamma = self.model.state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		s = np.sqrt(VarSI + VarxI + cov + eps)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		nc = len(self.model.N)
		lz = self.model.state['lz']
		m = gp.Model("qp")
		F = np.zeros(self.proj_dim)
		x = m.addMVar(shape=F.shape, vtype=GRB.CONTINUOUS, name="x")

		pbarI1 = Ibar1/N
		v = pbarI1 * (1-pbarI1)
		Az = np.diag(v)
		BAB = np.dot(np.dot(self.A.T, Az), self.A)
		cp = np.dot(np.dot(N, Az), self.A)


		m.setParam('OutputFlag', 0)
		m.setObjective(x @ BAB @ x + cp @ x, GRB.MAXIMIZE)

		M = np.dot(np.ones(nc), self.A)
		Ntest = np.array([ntest])

		# Add constraints
		m.addConstr(M @ x == Ntest, name="c")
		m.addConstr(0 <= x)
		m.setParam('NonConvex', 2)
		m.optimize()
		xx = x.X
		return np.int32(np.dot(self.A, xx))


class KG_grad:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.vacpol = vacpol
		self.S = params[0]
		self.M = params[1]


	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		x0 = np.random.rand(self.model.nc)
		x0 = np.int32(x0 / sum(x0) * ntest)
		state1 = self.model.state.copy()
		x = self.RGD(x0, state1, xvac)
		return x


	def RGD(self, x0, state0, xvac):
		ntest = state0['ntest']
		xk = x0
		for k in range(self.M):
			print('k')
			print('x: ' + str(xk))
			grad = np.zeros(xvac.shape)
			for i in range(self.S):
				grad += self.num_grad(xk, state0, xvac)
			grad /= self.S
			print(grad)
			xk = xk + 5*grad
			xk = np.max([np.zeros(self.model.nc), xk], axis=0)
			xk = simplex_projector_integer(xk, ntest)

		return xk

	def num_grad(self, x, state, xvac):
		diff = np.zeros(xvac.shape)
		input1 = x.copy()
		input2 = x.copy()
		for i in range(state['nc']):
			input1[i] = input1[i] + 1
			input2[i] = input2[i] - 1
			diff[i] = 0.5 * (self.sim_forward(input1, xvac, state) - self.sim_forward(input2, xvac, state))
		return diff


	def sim_forward(self, xtest, xvac, state):
		state = state.copy()
		N = state['N']
		pS = state['pS']
		pI = state['pI']
		pR = state['pR']
		nc = state['nc']
		beta = state['beta']
		gamma = state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		s = np.sqrt(VarSI + VarxI + cov + eps)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		nc = len(self.model.N)
		lz = self.model.state['lz']

		alpha_tz = lz * Ibar1
		beta_tz = lz * (N - Ibar1)

		# update the estimator for the beta-binomial
		bool = xtest < 0
		xtest[bool] = 0

		Ihat = [np.random.binomial(xtest[i], Ibar[i] / N[i]) for i in range(nc)]
		p_inf = (Ihat + alpha_tz) / (xtest + lz * N)

		# project the other dimensions back onto it
		sv = np.array(
			[simplex_projector(np.array([Sbar1[i] / N[i], Rbar1[i] / N[i]]), const=p_inf[i]) for i in range(nc)])
		return np.sum(-gamma * N * p_inf + beta * p_inf * np.max([((N-1) * sv[:,0] - xvac), np.zeros(xvac.shape)], axis=0))

class pure_exploration:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.vacpol = vacpol
		self.params = params

	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		x0 = np.array([np.random.randint(low=0, high=10000) for _ in range(self.model.nc)])
		x0 = np.int32(x0 / sum(x0) * ntest)
		while np.sum(x0) > ntest:
			j = np.random.randint(high = self.model.nc)
			x0[j] -= 1
		return x0

class full_test:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.vacpol = vacpol
		self.params = params

	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		x0 = self.model.N
		return x0

class REMBO_TS:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.proj_dim = np.int32(params[0])
		self.vacpol = vacpol
		self.A = np.random.rand(np.int32(self.model.nc), self.proj_dim)


	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		cs = []
		vacpol = self.vacpol.__copy__()
		Xtest = np.random.rand(self.model.nc, self.xsamps)
		Xtest = Xtest / np.sum(Xtest, axis=0)
		for k in range(self.xsamps):
			xtest = Xtest[:, k]

			ntest = self.model.state['ntest']
			state1 = self.model.state.copy()

			N = self.model.state['N']
			xtest = np.int32(N * xtest)
			pS = self.model.state['pS']
			pI = self.model.state['pI']
			pR = self.model.state['pR']
			beta = self.model.state['beta']
			gamma = self.model.state['gamma']
			eps = 1e-6

			# compute the necessary satistics
			Sbar = pS * N
			Ibar = pI * N
			Rbar = pR * N

			EISx = N * pI * ((N - 1) * pS - xvac)
			ESI = N * (N - 1) * pI * pS
			ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
			VarSI = ES2I2 - ESI ** 2
			VarxI = xvac ** 2 * (N * pI * (1 - pI))
			cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
			s = np.sqrt(VarSI + VarxI + cov + eps)
			mean = EISx

			# E[I S^x]
			term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

			# E[max(0,S-x)] = E[S] - E[min(S,x)]
			sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
			ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
				(Sbar - xvac) / sigma_susc)
			Emin = Sbar - ESx

			# Get the predicted states at t+1
			Sbar1 = ESx - term
			Ibar1 = (1 - gamma) * Ibar + term
			Rbar1 = Rbar + gamma * Ibar + Emin

			nc = len(self.model.N)
			lz = self.model.state['lz']

			pbarI1 = np.min([(1 - 10e-6) * np.ones(nc), Ibar1 / N + 10e-6], axis=0)

			Ihat = np.array([np.random.binomial(xtest[i], pbarI1[i]) for i in range(nc)])

			pI1 = (Ihat + lz * Ibar1) / (xtest + lz * N)

			sv = np.array([simplex_projector(np.array([Sbar1[i] / N[i], Rbar1[i] / N[i]]), const=pI1[i]) for i in
			               range(nc)])

			pS1 = sv[:, 0]
			pR1 = sv[:, 1]

			state1['pS'] = pS1
			state1['pI'] = pI1
			state1['pR'] = pR1

			model_temp = self.model.__copy__()
			model_temp.state = state1.copy()
			vacpol.update(model_temp, 0)
			c = np.sum(-gamma * N * pI1 + beta * pI1 * np.max([np.zeros(nc), (N - 1) * pS1 - vacpol.decision()]))
			cs.append(c)
		kstar = np.argmin(cs)
		return Xtest[:, kstar]

class KGstar:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.vacpol = vacpol



	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		N = self.model.state['N']
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		sigmaI = self.model.state['sigmaI']
		beta = self.model.state['beta']
		gamma = self.model.state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		VarI = VarSI + VarxI + cov+ eps
		s = np.sqrt(VarI)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		fI1 = Ibar/N

		VarpI = VarI / N**2

		betat = np.max([1/sigmaI - 1/VarpI, 1e-10 * np.ones(self.model.nc)], axis=0)
		nx = np.arange(1,100)
		sigma_til = np.array([1/betat - 1/(betat + xx * (1/(fI1 * (1-fI1)))) for xx in nx])

		c = []
		for i in range(self.model.nc):
			il = list(np.arange(0,self.model.nc))
			il.pop(i)
			pnext = np.max(pI[il])
			c.append(- np.abs(pI[i] - pnext / sigma_til[:,i]))

		f = c * stats.norm.cdf(c, 0, 1) + stats.norm.pdf(c, 0, 1)
		v = sigma_til * f.T
		vstar = v / nx.repeat(self.model.nc).reshape(nx.shape[0], self.model.nc)

		xstars = np.argmax(vstar, axis=0)
		xstars[xstars == 0] = 1

		vstars = np.max(vstar, axis=0)

		ii = np.argsort(vstars)
		decisions = xstars[ii]
		dcumul = np.cumsum(decisions)
		bools = dcumul < ntest
		decision = np.zeros(self.model.nc)
		decision[ii[bools]] = xstars[ii[bools]]

		if np.sum(decision) < ntest:
			leftover = ntest - np.sum(decision)
			decision += np.int32(leftover / self.model.nc)

		return decision


class KGstar_tradeoff:
	def __init__(self, model, params, vacpol):
		self.model = model
		self.vacpol = vacpol

	def update(self, model_new, params, vacpol):
		self.model = model_new
		self.vacpol = vacpol

	def decision(self, xvac):
		ntest = self.model.state['ntest']
		pcts = self.model.N / sum(self.model.N)
		xtest = self.pr * ntest * pcts

		ntest1 = np.floor(ntest - np.sum(xtest))

		N = self.model.state['N']
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		sigmaI = self.model.state['sigmaI']
		beta = self.model.state['beta']
		gamma = self.model.state['gamma']
		eps = 1e-6

		# compute the necessary satistics
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		EISx = N * pI * ((N - 1) * pS - xvac)
		ESI = N * (N - 1) * pI * pS
		ES2I2 = N * (N - 1) * pI * pS * (1 + (pI + pS) * (N - 2) + pI * pS * (N - 2) * (N - 3))
		VarSI = ES2I2 - ESI ** 2
		VarxI = xvac ** 2 * (N * pI * (1 - pI))
		cov = -2 * xvac * N * (N - 1) * pI * pS * (1 - 2 * pI)
		VarI = VarSI + VarxI + cov+ eps
		s = np.sqrt(VarI)
		mean = EISx

		# E[I S^x]
		term = (beta / N) * ((EISx) * stats.norm.cdf(mean / s) + s * stats.norm.pdf(mean / s))

		# E[max(0,S-x)] = E[S] - E[min(S,x)]
		sigma_susc = np.sqrt(N * pS * (1 - pS) + eps)
		ESx = (Sbar - xvac) * stats.norm.cdf((Sbar - xvac) / sigma_susc) + sigma_susc * stats.norm.pdf(
			(Sbar - xvac) / sigma_susc)
		Emin = Sbar - ESx

		# Get the predicted states at t+1
		Sbar1 = ESx - term
		Ibar1 = (1 - gamma) * Ibar + term
		Rbar1 = Rbar + gamma * Ibar + Emin

		fI1 = Ibar/N

		VarpI = VarI / N**2

		betat = np.max([1/sigmaI - 1/VarpI, 1e-10 * np.ones(self.model.nc)], axis=0)
		nx = np.arange(1,100)
		sigma_til = np.array([1/betat - 1/(betat + xx * (1/(fI1 * (1-fI1)))) + 1e-6 for xx in nx])

		c = []
		for i in range(self.model.nc):
			il = list(np.arange(0,self.model.nc))
			il.pop(i)
			pnext = np.max(pI[il])
			c.append(- np.abs(pI[i] - pnext / sigma_til[:,i]))

		f = c * stats.norm.cdf(c, 0, 1) + stats.norm.pdf(c, 0, 1)
		v = sigma_til * f.T
		vstar = v / nx.repeat(self.model.nc).reshape(nx.shape[0], self.model.nc)

		xstars = np.argmax(vstar, axis=0) + 1
		vstars = np.max(vstar, axis=0) + 1e-6
		decision = xstars / np.sum(xstars) * ntest1

		return decision
