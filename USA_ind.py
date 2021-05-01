import numpy as np
import USA_sim
import model as mod
import policy
import test_policy
import data_loader as dl


POP = dl.load_data('USA_DATA/state_population.obj')
FLOW = dl.load_data('USA_DATA/FLOW_STATE.obj')

# countyf = dl.load_data('USA_DATA/fips_county.obj')
# idx_dict = dl.load_data('USA_DATA/idx_dict2.obj')
betas = dl.load_data('USA_DATA/state_beta.obj')
nc = len(POP)
N = np.array(POP)
T = 20
vac_effic = 1
gamma = 0.3
alpha = 0.8
beta_ = np.array(betas)

def NVac_const(t):
	return nc * 100

def NTest_const(t):
	return 200

def NVac_t(t):
	a = np.zeros(T)
	tt = np.arange(0,T)
	start = 2
	a[start:] = 25*tt[start:] + 40
	return a[t]

def stochNVac_t(t):
	c=10
	np.random.seed(c)
	a = np.random.normal(50, 10, size=T)
	return np.int32(10*nc*t + 100*nc*a[t])

vacfun = stochNVac_t
testfun = NTest_const
D = 2
Ig = []
cc =[]
Icounties = []
for d in range(D):
	p_inf_ = 0.1
	p_rec_ = 0.001
	# generate initial states
	ii = 3
	p_inf = p_inf_ * np.random.rand(nc)
	p_inf[ii] = p_inf_
	p_rec = p_rec_ * np.ones(nc)
	p_susc = 1 - p_inf - p_rec

	env_state = {}
	env_state = {'N':N, 'beta_':beta_, 'vac_effic':vac_effic, 'gamma_':gamma, 'p_susc':p_susc, 'p_inf':p_inf, 'p_rec':p_rec, 'alpha_':alpha}
	env_state['nvac'] = vacfun(0)
	# vaccine policies

	# testing policy
	test_policy_names = ['prop_greedy']
	k = len(test_policy_names)
	sims = USA_sim.Simulator(env_state.copy(), nc, T)

	Models = mod.model(sims, nvac = vacfun(0), ntest=testfun(0))
	pname = 'DLA'
	policyd = policy.risk_adjusted_DLA
	param_list = [0.25]
	policy_r = policyd(Models, param_list)

	# test_policies = [test_policy.prop_allocate(Models[0],0), test_policy.prop_allocate(Models[1],0)]
	test_policies = test_policy.prop_greedy_trade(Models,pr=0.75)
	cum_list = np.zeros(T - 1)
	cumrew = 0
	I_list = np.zeros(T-1)

	for t in range(T-1):
		print('t = ' +str(t))
		# new information has been recieved and belief models updated
		policy_r.update(Models,0)
		xvac_dec = policy_r.decision(t)
		test_policies.update(Models,xvac_dec, policy_r)
		xtest_dec = test_policies.decision()

		# environmental transition to t+1
		pbar = sims.forward_one_step(xvac=xvac_dec, nvac=vacfun(t), t=t)

		# model transitions to t+1
		Models.forward_one_step(xvac_dec,xtest_dec,vacfun(t+1), testfun(t+1))

		cumrew += pbar
		cum_list[t] = cumrew
		I_list[t] = pbar

	Ig.append(I_list)
	cc.append(cum_list)
	Icounties.append(sims.I)

tt = np.arange(0, T)
vacs = np.zeros(tt.shape)
for rr in range(len(tt)):
	vacs[rr] = vacfun(rr)


dl.save_data(np.array(Ig), 'USA_DATA/Is_' + str(pname) + '.obj')
dl.save_data(cc, 'USA_DATA/DATA_' + str(pname) + '.obj')
dl.save_data(N, 'USA_DATA/NDATA_' + str(pname) + '.obj')
dl.save_data(vacs, 'USA_DATA/vacs_' + str(pname) + '.obj')
dl.save_data(Icounties, 'USA_DATA/counties_' + str(pname) + '.obj')

