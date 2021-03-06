import numpy as np
from Controller import *
from Simulator import *
import cost_object
from utility import *


def run_sample_path(hyperparameters, stochastics, vac_policy, test_policy, vparam, tparam, perfect=False):
	# run a sample path of the system
	np.random.seed(stochastics[5])
	N = hyperparameters[14]
	nc = hyperparameters[0]
	T = hyperparameters[1]
	vac_fun = stochastics[1]
	test_fun = stochastics[2]

	I = hyperparameters[10] * N * stochastics[4]
	R = hyperparameters[11] * N
	# true number of initial susceptibles
	S = N - I - R
	dI = I.copy()


	# initial environment state
	env_state = {'S': S, 'I': I, 'R': R, 'dI': dI, 'beta_': stochastics[0], 'gamma_': hyperparameters[12],
	             'bw': hyperparameters[15], 'locs': hyperparameters[16],
	             'alpha_': hyperparameters[13], 'xi': hyperparameters[2], 'N': N, 'nc': hyperparameters[0],
	             'T': hyperparameters[1], 'a': hyperparameters[4], 'b': hyperparameters[5], 'cc': hyperparameters[6],
	             'dd': hyperparameters[7], 'fp': hyperparameters[8], 'fn': hyperparameters[9],
	             'vac_fun': stochastics[1], 'test_fun': stochastics[2], 'Movers': stochastics[3]}
	env_funs = [vac_fun, test_fun]
	sim = simulator_model(env_state, env_funs)
	# sim.tracker.getI0(dI)
	# generate initial state

	# initialize belief state
	pI = 0.01 * np.ones(nc)
	pS = 1 - pI
	pR = np.zeros(nc)



	gamma_ = hyperparameters[12]
	xi_ = hyperparameters[2]
	sigmaI = pI * (1-pI)

	state = {'pS': pS, 'pI': pI, 'pR': pR, 'N': N, 'nc': nc, 'beta':  stochastics[0][:,0], 'gamma': gamma_,
	         'nvac': vac_fun(0), 'ntest': test_fun(0), 't': 0, 'lz': hyperparameters[3],
	         'bw_approx': hyperparameters[17], 'locs': hyperparameters[16], 'xi': xi_, 'sigmaI':sigmaI}
	if perfect:
		controller_funs = [sim.getI]
	else:
		controller_funs = [sim.obs_fun]

	model = controller(state, controller_funs)

	# initialize cost object
	CO = cost_object.single_cost_object()
	CO.update(np.sum(dI))

	# initialize vaccine and test policies
	vac_pol = vac_policy(model, vparam)
	test_pol = test_policy(model, tparam, vac_pol.__copy__())

	xvaclist = []
	Ilist = []

	xvac = vac_pol.decision()
	xtest = test_pol.decision(xvac)
	xvaclist.append(xvac)
	Ilist.append(N * model.state['pI'])
	for t in np.arange(1, T - 1):
		# print('t = '+str(t))
		# step environment forward to t+1
		c = sim.forward_one_step(xvac, t)
		CO.update(c)

		nvac = vac_fun(t)
		ntest = test_fun(t)

		# step controller forward to t+1
		model.forward_one_step(xvac, xtest, nvac, ntest)

		# make new vaccine decision
		vac_pol.update(model, vparam)
		xvac = vac_pol.decision()
		test_pol.update(model, tparam, vac_pol.__copy__())
		xtest = test_pol.decision(xvac)
		xvaclist.append(xvac)
		Ilist.append(N * model.state['pI'])


	c = sim.forward_one_step(xvac, T - 1)
	CO.update(c)
	# sim.tracker.plot(0)

	return CO, xvaclist, Ilist


def run_sample_path2(hyperparameters, stochastics, vac_policy, test_policy, vparam, tparam, m, perfect=False):
	# run a sample path of the system
	np.random.seed(stochastics[5])
	N = hyperparameters[14]
	nc = hyperparameters[0]
	T = hyperparameters[1]
	vac_fun = stochastics[1]
	test_fun = stochastics[2]

	I = hyperparameters[10] * N * stochastics[4]
	R = hyperparameters[11] * N
	# true number of initial susceptibles
	S = N - I - R
	dI = I.copy()


	# initial environment state
	env_state = {'S': S, 'I': I, 'R': R, 'dI': dI, 'beta_': stochastics[0], 'gamma_': hyperparameters[12],
	             'bw': hyperparameters[15], 'locs': hyperparameters[16],
	             'alpha_': hyperparameters[13], 'xi': hyperparameters[2], 'N': N, 'nc': hyperparameters[0],
	             'T': hyperparameters[1], 'a': hyperparameters[4], 'b': hyperparameters[5], 'cc': hyperparameters[6],
	             'dd': hyperparameters[7], 'fp': hyperparameters[8], 'fn': hyperparameters[9],
	             'vac_fun': stochastics[1], 'test_fun': stochastics[2], 'Movers': stochastics[3]}
	env_funs = [vac_fun, test_fun]
	sim = simulator_model(env_state, env_funs)
	# sim.tracker.getI0(dI)
	# generate initial state

	# initialize belief state
	pI = 0.01 * np.ones(nc)
	pS = 1 - pI
	pR = np.zeros(nc)



	gamma_ = hyperparameters[12]
	xi_ = hyperparameters[2]
	sigmaI = pI * (1-pI)

	state = {'pS': pS, 'pI': pI, 'pR': pR, 'N': N, 'nc': nc, 'beta':  stochastics[0][:,0], 'gamma': gamma_,
	         'nvac': vac_fun(m), 'ntest': test_fun(0), 't': 0, 'lz': hyperparameters[3],
	         'bw_approx': hyperparameters[17], 'locs': hyperparameters[16], 'xi': xi_, 'sigmaI':sigmaI}
	if perfect:
		controller_funs = [sim.getI]
	else:
		controller_funs = [sim.obs_fun]

	model = controller(state, controller_funs)

	# initialize cost object
	CO = cost_object.single_cost_object()
	CO.update(np.sum(dI))

	# initialize vaccine and test policies
	vac_pol = vac_policy(model, vparam)
	test_pol = test_policy(model, tparam, vac_pol.__copy__())

	xvaclist = []
	Ilist = []

	xvac = vac_pol.decision()
	xtest = test_pol.decision(xvac)
	xvaclist.append(xvac)
	Ilist.append(N * model.state['pI'])
	for t in np.arange(1, T - 1):
		# print('t = '+str(t))
		# step environment forward to t+1
		c = sim.forward_one_step(xvac, t)
		CO.update(c)

		nvac = vac_fun(m)
		ntest = test_fun(t)

		# step controller forward to t+1
		model.forward_one_step(xvac, xtest, nvac, ntest)

		# make new vaccine decision
		vac_pol.update(model, vparam)
		xvac = vac_pol.decision()
		test_pol.update(model, tparam, vac_pol.__copy__())
		xtest = test_pol.decision(xvac)
		xvaclist.append(xvac)
		Ilist.append(N * model.state['pI'])


	c = sim.forward_one_step(xvac, T - 1)
	CO.update(c)
	# sim.tracker.plot(0)

	return CO, xvaclist, Ilist