import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.stats as stats
import utility
from scipy.spatial.distance import squareform, pdist
from utility import proj as gradproj
import matplotlib.pyplot as plt
class null_policy:
	# This policy allocates zero vaccines to any zones (This gives a baseline)
	def __init__(self, model, params):
		# params is an empty list
		self.model = model
		self.params = params

	def update(self, model, new_params):
		# new_params is an empty list
		self.model = model

	def __copy__(self):
		return null_policy(self.model, self.params)

	def decision(self):
		decision = np.zeros(self.model.nc)
		return decision

class even_policy:
	# This policy allocates zero vaccines to any zones (This gives a baseline)
	def __init__(self, model, params):
		# params is an empty list
		self.model = model
		self.params = params

	def update(self, model, new_params):
		# new_params is an empty list
		self.model = model

	def __copy__(self):
		return even_policy(self.model, self.params)

	def decision(self):

		nvac = self.model.state['nvac']
		pS = self.model.state['pS']
		N = self.model.state['N']
		Sbar = N * pS
		if np.sum(Sbar) < nvac:
			decision = Sbar
		else:

			pcts = Sbar / sum(Sbar)
			decision = nvac * pcts
		return decision

class prop_policy:
	# This policy allocates zero vaccines to any zones (This gives a baseline)
	def __init__(self, model, params):
		# params is an empty list
		self.model = model
		self.params = params

	def update(self, model, new_params):
		# new_params is an empty list
		self.model = model

	def __copy__(self):
		return prop_policy(self.model, self.params)

	def decision(self):

		nvac = self.model.state['nvac']
		pS = self.model.state['pS']
		N = self.model.state['N']
		Sbar = N * pS
		if np.sum(Sbar) < nvac:
			decision = Sbar
		else:
			pcts = N / sum(N)
			decision = np.floor(nvac * pcts)
		return decision

class constant_policy:
	def __init__(self, model, params):
		# params is an empty list
		self.model = model
		self.params = params

	def update(self, model, new_params):
		# new_params is an empty list
		self.model = model
		self.params = new_params

	def __copy__(self):
		return constant_policy(self.model, self.params)

	def decision(self):
		decision = self.model.N
		return decision

class susc_allocate:
	def __init__(self, model,param_list):
		self.model = model
		self.params = param_list
		self.a = param_list[0]
		self.k = param_list[1]

	def __copy__(self):
		return susc_allocate(self.model, self.params)

	def update(self, model_new,_):
		self.model = model_new

	def decision(self):
		nvac = self.model.state['nvac']
		# print('nvac = ' + str(nvac))
		pS = self.model.state['pS']
		N = self.model.state['N']
		Sbar = N * pS
		if np.sum(Sbar) < nvac:
			decision = Sbar
		else:
			pS = self.model.state['pS']
			pt = (1 / (1 + np.exp(-self.k * pS))) ** self.a
			nc = self.model.nc
			decision = nvac * (pt / np.sum(pt))
		# xi = np.floor(nvac / nc)
		# rem = np.int32(nvac % nc)
		# x = xi * np.ones(nc)
		# if rem > 0:
		# 	x[:rem] += 1
		# print('allocations = ' +str(decision))
		return decision

class Sampled_greedy:
	# This policy implements a greedy algorithm with respect to a sample of the belief state
	def __init__(self, model, params):
		self.model = model
		self.params = params

	def __copy__(self):
		return Sampled_greedy(self.model, self.params)

	def update(self, model, params):
		self.model = model
		self.params = params

	def decision(self):
		beta_ = self.model.beta
		N = self.model.N
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		V = np.array([np.random.multinomial(N[i],[pS[i],pI[i],pR[i]]) for i in range(len(N))])
		Shat = V[:, 0]
		Ihat = V[:, 1]
		Rhat = V[:, 2]
		nvac = self.model.state['nvac']
		nc = self.model.nc
		pihat = Ihat / N
		c = beta_ * pihat
		m = gp.Model("iP")
		# Create variables
		x = m.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

		m.setParam('OutputFlag', 0)
		m.setObjective(c @ x, GRB.MAXIMIZE)
		m.addConstr(np.ones(nc) @ x <= nvac, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= Shat)
		m.optimize()
		return x.X

class risk_greedy:
	# This risk adjusted CFA
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]

	def __copy__(self):
		return Sampled_greedy(self.model, self.params)

	def update(self, model, params):
		self.model = model
		self.params = params

	def decision(self):
		beta_ = self.model.beta
		N = self.model.N
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']

		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01
		Salpha = np.max([np.zeros(self.model.nc), mean - std * stats.norm.ppf(self.alpha)], axis=0)

		nvac = self.model.state['nvac']
		nc = self.model.nc
		c = beta_ * pI
		m = gp.Model("iP")
		# Create variables
		x = m.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

		m.setParam('OutputFlag', 0)
		m.setObjective(c @ x, GRB.MAXIMIZE)
		m.addConstr(np.ones(nc) @ x <= nvac, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= Salpha)
		m.optimize()
		return x.X


class multidim_greedy:
	# This policy implements a greedy algorithm with respect to a sample of the belief state
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]
		self.kappa = params[1]
		self.FLOW = np.exp(-0.5 * pdist(self.model.state['locs'] / self.model.state['bw_approx'], 'sqeuclidean'))
		self.FLOW = squareform(self.FLOW)
		self.FLOW = np.min(
			[0.8 * np.ones(self.FLOW.shape), np.max([0.001 * np.ones(self.FLOW.shape), self.FLOW], axis=0)],
			axis=0)

	def __copy__(self):
		return multidim_greedy(self.model, self.params)

	def update(self, model, params):
		self.model = model
		self.params = params

	def decision(self):
		M = self.FLOW
		beta_ = self.model.beta
		N = self.model.N
		pS = self.model.state['pS']
		pI = self.model.state['pI']
		pR = self.model.state['pR']
		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean + std * stats.norm.ppf(self.alpha)], axis=0)

		nvac = self.model.state['nvac']
		nc = self.model.nc

		Identity = np.eye(M.shape[0])
		d = np.dot((Identity + self.kappa * M), pI*N)

		c = beta_ * d
		m = gp.Model("iP")
		# Create variables
		x = m.addMVar(shape=nc, vtype=GRB.INTEGER, name="x")

		m.setParam('OutputFlag', 0)
		m.setObjective(c @ x, GRB.MAXIMIZE)
		m.addConstr(np.ones(nc) @ x <= nvac, name="c")
		m.addConstr(0 <= x)
		m.addConstr(x <= Salpha)
		m.optimize()
		return x.X

class sampled_DLA:
	def __init__(self, model, params):
		self.model = model
		self.params = params

	def __copy__(self):
		return sampled_DLA(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		xstar = self.solve_Quad()
		return xstar

	def solve_Quad(self):
		N = self.model.N
		beta = self.model.beta
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']


		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		V = np.array([np.random.multinomial(N[i], [pS[i], pI[i], pR[i]]) for i in range(len(N))])
		Shat = V[:, 0]
		Ihat = V[:, 1]
		Rhat = V[:, 2]

		pIhat = Ihat/N

		if np.sum(Shat) < nvac:
			decision = Shat
		else:
			if nc <52:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
				m.setParam('OutputFlag', 0)
				m.setParam('TimeLimit', 10)
				A1 = -beta * pIhat
				A2 = (1-gamma) * N * pIhat + beta*(N-1)*pS*pI
				K1 = (beta * pIhat) - 1
				K2 = (1-(beta*pIhat)) * Shat

				p = (1-gamma) * A1 + (beta/N) * A1 * K2 + (beta/N) * A2 * K1
				q = -(beta/N) * A2
				Q00 = np.diag((beta/N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta/N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				c = np.hstack([p,q])
				m.setObjective(x @ Q @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Shat, K2])
				m.addConstr(G @ x <= R)
				m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
			else:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
				m.setParam('TimeLimit', 10)
				m.setParam('OutputFlag', 0)
				A1 = -beta * pIhat
				A2 = (1 - gamma) * N *pIhat+ beta * (N - 1) * pS * pI
				K1 = (beta * pIhat) - 1
				K2 = (1-(beta*pIhat)) * Shat

				p = (1 - gamma) * A1 + (beta / N) * A1 * K2 + (beta / N) * A2 * K1
				q = -(beta / N) * A2
				Q00 = np.diag((beta / N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta / N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				# Qproj = np.zeros([2*nc, 2*nc])
				# Qproj[nc:, :nc] = Q01
				# Qproj[:nc, nc:] = Q01
				e,v = np.linalg.eig(Q)
				e = np.max([1e-10*np.ones(2*nc), e], axis=0)

				# e = np.zeros(2*nc)
				Qproj = np.dot(np.dot(v.T, np.diag(e)), v)
				c = np.hstack([p, q])
				m.setObjective(x @ Qproj @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Shat, K2])
				m.addConstr(G @ x <= R)
				# m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
		return decision



class risk_adjusted_DLA:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]

	def __copy__(self):
		return risk_adjusted_DLA(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		xstar = self.solve_Quad()
		return xstar

	def solve_Quad(self):
		N = self.model.N
		beta = self.model.beta
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']


		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean - std * stats.norm.ppf(self.alpha)], axis=0)

		if np.sum(Salpha) < nvac:
			decision = Salpha
		else:
			if nc <24:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
				m.setParam('OutputFlag', 0)
				m.setParam('TimeLimit', 10)
				A1 = -beta * pI
				A2 = (1-gamma) * N * pI + beta*(N-1)*pS*pI
				K1 = (beta * pI) - 1
				K2 = (1-(beta*pI)) * Salpha

				p = (1-gamma) * A1 + (beta/N) * A1 * K2 + (beta/N) * A2 * K1
				q = -(beta/N) * A2
				Q00 = np.diag((beta/N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta/N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				c = np.hstack([p,q])
				m.setObjective(x @ Q @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Salpha, K2])
				m.addConstr(G @ x <= R)
				m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
			else:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
				m.setParam('OutputFlag', 0)
				m.setParam('TimeLimit', 10)

				A1 = -beta * pI
				A2 = (1 - gamma) * N * pI + beta * (N - 1) * pS * pI
				K1 = (beta * pI) - 1
				K2 = (1-(beta*pI)) * Salpha

				p = (1 - gamma) * A1 + (beta / N) * A1 * K2 + (beta / N) * A2 * K1
				q = -(beta / N) * A2
				Q00 = np.diag((beta / N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta / N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				# Qproj = np.zeros([2*nc, 2*nc])
				# Qproj[nc:, :nc] = Q01
				# Qproj[:nc, nc:] = Q01
				e,v = np.linalg.eig(Q)
				e = np.max([1e-10*np.ones(2*nc), e], axis=0)

				# e = np.zeros(2*nc)
				Qproj = np.dot(np.dot(v.T, np.diag(e)), v)
				c = np.hstack([p, q])
				m.setObjective(x @ Qproj @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac+200])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Salpha, K2])
				m.addConstr(G @ x <= R)
				# m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
		return decision





class simulated_horizon_DLA:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.H = params[0]
		self.a = 50
		self.k = 10

	def __copy__(self):
		return simulated_horizon_DLA(self.model, self.params)

	def update(self, model, _):
		self.model = model

	def decision(self):
		nc = self.model.nc
		N = self.model.N
		nvac = self.model.state['nvac']
		m = gp.Model("iP")
		x = m.addMVar(shape=(self.H+1) * nc, vtype=GRB.INTEGER, name="x")

		state0 = self.model.state.copy()
		temp_state = state0.copy()
		slist = [state0]
		ctp = [temp_state['beta'] * temp_state['pI']]
		stp = [temp_state['pS']*N]
		G = np.ones([nc,1])
		Nvac = [nvac]
		for tp in range(self.H):
			temp_state = self.simulate(temp_state)
			slist.append(temp_state)
			ctp.append(temp_state['beta'] * temp_state['pI'])
			G = np.vstack(
				[np.hstack([G, np.zeros([G.shape[0], 1])]), np.hstack([np.zeros([nc, G.shape[1]]), np.ones([nc, 1])])])
			Nvac.append(nvac)
			stp.append(temp_state['pS']*N)

		c = np.array(ctp).flatten()
		s = np.array(stp).flatten()
		Nvac = np.array(Nvac)
		m.setParam('OutputFlag', 0)
		m.setObjective(c @ x, GRB.MAXIMIZE)

		m.addConstr(G.T @ x <= Nvac, name="c")
		m.addConstr(x <= s)
		m.addConstr(0 <= x)

		m.optimize()
		xx = x.X
		return xx[:nc]


	def simulate(self, state):
		eps = 1e-6
		beta = state['beta']
		gamma = state['gamma']
		pS = state['pS']
		pI = state['pI']
		pR = state['pR']
		N = state['N']
		nc = state['nc']
		Sbar = pS * N
		Ibar = pI * N
		Rbar = pR * N

		xvac = self.decision_sim(state)

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

		state1 = state.copy()
		tempS = Sbar1 / N
		tempI = Ibar1 / N
		tempR = Rbar1 / N

		proj = np.array([utility.simplex_projector(np.array([tempS[i],tempI[i],tempR[i]]), 1) for i in range(len(Sbar1))])

		state1['pS'] = proj[:,0]
		state1['pI'] = proj[:,1]
		state1['pR'] = proj[:,2]
		return state1

	def decision_sim(self, state):
		pS = state['pS']
		pt = (1 / (1 + np.exp(-self.k * pS))) ** self.a
		nvac = self.model.state['nvac']
		nc = self.model.nc
		x = nvac * (pt / np.sum(pt))
		return x

class linear_approx_2step:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]

	def __copy__(self):
		return linear_approx_2step(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		xstar = self.solve_lp()
		return xstar

	def solve_lp(self):
		N = self.model.N
		beta = self.model.beta
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']

		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean + std * stats.norm.ppf(self.alpha)], axis=0)


		if np.sum(Salpha) < nvac:
			decision = Salpha
		else:
			A = -beta * pI
			B = -gamma * pI + beta * pI * (N-1) * pS
			C = -(N - beta) * pI
			D = (N-beta) * (N-1) * pI * pS

			K1 = (beta * pI) - 1
			K2 = (1 - (beta * pI)) * Salpha

			m = gp.Model("qp")
			# Create variables
			x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

			# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
			m.setParam('OutputFlag', 0)

			p = -gamma * A + (beta/N) * B * C + (beta/N) * A * D
			q = (beta/N) * B

			c = np.hstack([p,q])
			m.setObjective(-c @ x, GRB.MAXIMIZE)

			M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
			# Nvac = np.array([nvac, nvac+80])
			Nvac = np.array([nvac, nvac])
			# Add constraints
			m.addConstr(M @ x <= Nvac, name="c")
			m.addConstr(0 <= x)
			G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
			R = np.hstack([Salpha, K2])
			m.addConstr(G @ x <= R)
			m.optimize()
			xx = x.X
			decision = xx[:nc]

		return decision


class fairness_risk_adjusted_DLA:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]
		self.rho_vac = params[1]

	def __copy__(self):
		return risk_adjusted_DLA(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		nvac = self.model.state['nvac']
		pcts = self.model.N / sum(self.model.N)
		xrho = self.rho_vac * nvac * pcts

		nvac1 = np.floor(nvac - np.sum(xrho))

		xstar = self.solve_Quad(nvac1)
		x = xstar + xrho
		return x

	def solve_Quad(self, nvac):
		N = self.model.N
		beta = self.model.beta
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']




		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean + std * stats.norm.ppf(self.alpha)], axis=0)

		if np.sum(Salpha) < nvac:
			decision = Salpha
		else:
			if nc <24:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
				m.setParam('OutputFlag', 0)
				m.setParam('TimeLimit', 10)
				A1 = -beta * pI
				A2 = (1-gamma) * N * pI + beta*(N-1)*pS*pI
				K1 = (beta * pI) - 1
				K2 = (1-(beta*pI)) * Salpha

				p = (1-gamma) * A1 + (beta/N) * A1 * K2 + (beta/N) * A2 * K1
				q = -(beta/N) * A2
				Q00 = np.diag((beta/N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta/N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				c = np.hstack([p,q])
				m.setObjective(x @ Q @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Salpha, K2])
				m.addConstr(G @ x <= R)
				m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
			else:
				m = gp.Model("qp")
				# Create variables
				x = m.addMVar(shape=2 * nc, vtype=GRB.INTEGER, name="x")

				# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x

				m.setParam('OutputFlag', 0)
				m.setParam('TimeLimit', 10)

				A1 = -beta * pI
				A2 = (1 - gamma) * N * pI + beta * (N - 1) * pS * pI
				K1 = (beta * pI) - 1
				K2 = (1-(beta*pI)) * Salpha

				p = (1 - gamma) * A1 + (beta / N) * A1 * K2 + (beta / N) * A2 * K1
				q = -(beta / N) * A2
				Q00 = np.diag((beta / N) * A1 * K1)
				Q01 = np.diag(-0.5 * (beta / N) * A1)
				Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
				# Qproj = np.zeros([2*nc, 2*nc])
				# Qproj[nc:, :nc] = Q01
				# Qproj[:nc, nc:] = Q01
				e,v = np.linalg.eig(Q)
				e = np.max([1e-10*np.ones(2*nc), e], axis=0)

				# e = np.zeros(2*nc)
				Qproj = np.dot(np.dot(v.T, np.diag(e)), v)
				c = np.hstack([p, q])
				m.setObjective(x @ Qproj @ x + c @ x, GRB.MINIMIZE)

				M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
				# Nvac = np.array([nvac, nvac+80])
				Nvac = np.array([nvac, nvac])
				# Add constraints
				m.addConstr(M @ x <= Nvac, name="c")
				m.addConstr(0 <= x)
				G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([-K1 * np.eye(nc), np.eye(nc)])])
				R = np.hstack([Salpha, K2])
				m.addConstr(G @ x <= R)
				# m.setParam('NonConvex', 2)
				m.optimize()
				xx = x.X
				decision = xx[:nc]
		return decision


class nonlinear_solve:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.H = np.int32(params[0])
		self.K = params[1]
		self.alpha = params[2]

	def __copy__(self):
		return nonlinear_solve(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		pS = self.model.state['pS']
		N = self.model.state['N']
		Sbar = N * pS
		nvac = self.model.state['nvac']
		if np.sum(Sbar) < nvac:
			decision = Sbar
		else:
			pcts = self.model.N / sum(self.model.N)
			xinit = [nvac * pcts for h in range(self.H)]
			# print('even:  ' + str(nvac * pcts))
			xstar = self.solve_nonlin(xinit, nvac)
			# print('xstar:  ' + str(xstar[:self.model.nc]))
			# print('beta = '+str(self.model.beta))
			decision = xstar[:self.model.nc]
		return decision

	def solve_nonlin(self, xinit, nvac):
		K = self.K
		nc = self.model.nc
		state = self.model.state.copy()
		kk = np.arange(1, K + 1)
		gg = 0.2
		c = 0.1
		# eta = np.max([0.01 * np.ones(eta.shape), hh/(eta+hh)], axis=0)
		ck = c / (kk) ** gg
		xinit = np.array(xinit).flatten()
		xtune1 = xinit.copy()
		xtune_t = xtune1.copy()
		mt = 0
		vt = 0
		b1 = 0.9
		b2 = 0.99
		eps = 10e-8
		g2 = 0
		lr = 5

		et = 0.8
		tlist = [list(xtune1)]

		# print('tunable1 = ' + str(xtune1))
		F = []
		for k in range(K):
			np.random.seed(None)
			# print('###  k = ' + str(k) + '  ####')

			vk = np.random.rand(xtune1.shape[0])
			hk = 2 * np.int32(vk > 0.5) - 1

			xtune_t = xtune1.copy()
			tune_tn1 = xtune_t - ck[k] * hk
			tune_t1 = xtune_t + ck[k] * hk
			tune_tn1 = np.array([gradproj(tune_tn1[(nc * i):(nc * (i + 1))], nvac) for i in range(self.H)]).flatten()
			tune_t1 = np.array([gradproj(tune_t1[(nc * i):(nc * (i + 1))], nvac) for i in range(self.H)]).flatten()

			F1 = self.simulator(state, tune_t1, self.H)
			Fn1 = self.simulator(state, tune_tn1, self.H)

			dG = (F1 - Fn1) / (2 * ck[k]) * hk
			grad_squared = dG * dG
			mt = b1 * mt + (1 - b1) * dG
			vt = b2 * vt + (1 - b2) * grad_squared
			mhat = mt / (1 - b1 ** (k + 1))
			vhat = vt / (1 - b2 ** (k + 1))
			alpha = lr / (np.sqrt(vhat) + eps)
			alphaG = alpha * mhat
			xtune1 = xtune_t - alphaG
			xtune1 = np.array([gradproj(xtune1[(nc * i):(nc * (i + 1))], nvac) for i in range(self.H)]).flatten()

			# print('tunable1 = ' + str(xtune1))
			# print('alpha = ' + str(alpha))
			# print('ck = ' + str(ck[k]))
			tlist.append(list(xtune1))

			F.append(F1)

		return xtune1

	def simulator(self, statet, x, T):
		# x comes in as a T*nc vector

		nc = statet['nc']
		N = statet['N']
		state = statet.copy()
		pSa = state['pS']
		pRa = state['pR']

		meanS = N * pSa

		std = np.sqrt(N * pSa * (1 - pSa))
		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), meanS - std * stats.norm.ppf(self.alpha)], axis=0)
		state['pS'] = Salpha/N


		xl = [x[(nc * i):(nc * (i + 1))] for i in range(T)]
		beta = state['beta']

		gamma = state['gamma']

		cost = []
		for t in range(T):
			pS = state['pS'].copy()
			pI = state['pI'].copy()
			pR = state['pR'].copy()
			N = state['N']
			Sx = np.max([np.zeros(nc), N * pS - xl[t]], axis=0)
			S = Sx - (beta) * pI * Sx
			I = (1 - gamma) * N * pI + (beta) * pI * Sx
			R = N * pR + gamma * N * pI + np.min([N * pS, xl[t]], axis=0)
			state['pS'] = S / N
			state['pI'] = I / N
			state['pR'] = R / N
			cost.append(np.sum(I - N * pI))
		cumuls = np.cumsum(cost)
		return np.cumsum(cumuls)[T - 1]




class risk_DLA_prime:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]

	def __copy__(self):
		return risk_DLA_prime(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		nvac = self.model.state['nvac']
		pcts = self.model.N / sum(self.model.N)
		# print('even:  ' + str(nvac * pcts))
		xstar = self.solve_Quad()
		# print('xstar:  ' + str(xstar[:self.model.nc]))
		# print('sum = ' + str(xstar.sum()))
		return xstar

	def solve_Quad(self):
		N = self.model.N
		beta = self.model.beta

		xi = self.model.state['xi']
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']
		Ibar = pI * N

		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean - std * stats.norm.ppf(self.alpha)], axis=0)
		# Salpha = np.min([N, Salpha], axis=0)


		if np.sum(Salpha) < nvac:
			decision = Salpha
		else:
			m = gp.Model("qp")
			# Create variables
			x = m.addMVar(shape=2 * nc, vtype=GRB.CONTINUOUS, name="x")

			# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
			m.setParam('OutputFlag', 0)
			m.setParam('TimeLimit', 20)

			A = -(beta/N)* Ibar
			B = (1-gamma)*Ibar + (beta/N)*Ibar*Salpha
			C = (beta/N)*Ibar - 1
			D = Salpha - (beta/N)*Ibar *Salpha

			p = (1-gamma)*A + (beta/N)*A*D + (beta/N)*B*C
			q = -(beta/N) *B

			Q00 = np.zeros([nc,nc])
			Q01 = np.zeros([nc,nc])
			np.fill_diagonal(Q00, (beta/N) * A * C)
			np.fill_diagonal(Q01, -(beta/N)*A)

			K1 = np.eye(nc)
			np.fill_diagonal(K1,-C)

			Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])
			c = np.hstack([p,q])
			m.setObjective(x @ Q @ x + c @ x, GRB.MINIMIZE)

			M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
			# Nvac = np.array([nvac, nvac+80])
			Nvac = np.array([nvac, nvac])
			# Add constraints
			m.addConstr(M @ x <= Nvac, name="c")
			m.addConstr(0 <= x)
			G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([K1, np.eye(nc)])])
			R = np.hstack([Salpha, D])
			m.addConstr(G @ x <= R)
			m.setParam('NonConvex', 2)
			m.optimize()
			xx = x.X
			decision = xx[:nc]

		return decision


class projectionDLA:
	def __init__(self, model, params):
		self.model = model
		self.params = params
		self.alpha = params[0]

	def __copy__(self):
		return risk_DLA_prime(self.model, self.params)

	def update(self, model_new, _):
		self.model = model_new

	def decision(self):
		nvac = self.model.state['nvac']
		pcts = self.model.N / sum(self.model.N)
		# print('even:  ' + str(nvac * pcts))
		xstar = self.solve_Quad()
		# print('xstar:  ' + str(xstar[:self.model.nc]))
		# print('sum = ' + str(xstar.sum()))
		return xstar

	def solve_Quad(self):
		N = self.model.N
		beta = self.model.beta

		xi = self.model.state['xi']
		gamma = self.model.gamma
		nc = self.model.nc
		state1 = self.model.state.copy()

		pS = state1['pS']
		pI = state1['pI']
		pR = state1['pR']
		Ibar = pI * N

		nvac = self.model.state['nvac']

		mean = N * pS
		std = np.sqrt(N * pS * (1 - pS))

		tt = std < 0.01
		std[tt] = 0.01

		Salpha = np.max([np.zeros(self.model.nc), mean - std * stats.norm.ppf(self.alpha)], axis=0)
		Salpha = np.min([N, Salpha], axis=0)


		if np.sum(Salpha) < nvac:
			decision = Salpha
		else:
			m = gp.Model("qp")
			# Create variables
			x = m.addMVar(shape=2 * nc, vtype=GRB.CONTINUOUS, name="x")

			# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
			m.setParam('OutputFlag', 0)
			m.setParam('TimeLimit', 20)

			A = -(beta/N)* Ibar
			B = (1-gamma)*Ibar + (beta/N)*Ibar*Salpha
			C = (beta/N)*Ibar - 1
			D = Salpha - (beta/N)*Ibar *Salpha

			p = (1-gamma)*A + (beta/N)*A*D + (beta/N)*B*C
			q = -(beta/N) *B

			Q00 = np.zeros([nc,nc])
			Q01 = np.zeros([nc,nc])
			np.fill_diagonal(Q00, (beta/N) * A * C)
			np.fill_diagonal(Q01, -(beta/N)*A)



			K1 = np.eye(nc)
			np.fill_diagonal(K1,-C)

			Q = np.vstack([np.hstack([Q00, Q01]), np.hstack([Q01, np.zeros([nc, nc])])])

			e, v = np.linalg.eig(Q)
			e = e.clip(min=1e-10)

			# e = np.zeros(2*nc)
			Qproj = np.dot(np.dot(v.T, np.diag(e)), v)




			c = np.hstack([p,q])
			m.setObjective(x @ Qproj @ x + c @ x, GRB.MINIMIZE)

			M = np.vstack([np.hstack([np.ones(nc), np.zeros(nc)]), np.hstack([np.zeros(nc), np.ones(nc)])])
			# Nvac = np.array([nvac, nvac+80])
			Nvac = np.array([nvac, nvac])
			# Add constraints
			m.addConstr(M @ x <= Nvac, name="c")
			m.addConstr(0 <= x)
			G = np.vstack([np.hstack([np.eye(nc), np.zeros([nc, nc])]), np.hstack([K1, np.eye(nc)])])
			R = np.hstack([Salpha, D])
			m.addConstr(G @ x <= R)
			m.optimize()
			xx = x.X
			decision = xx[:nc]

		return decision
