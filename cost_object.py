import numpy as np
import matplotlib.pyplot as plt

class single_cost_object:
	def __init__(self):
		self.inst_list = []
		self.cumul = 0
		self.cumul_list = []

	def update(self,c):
		self.inst_list.append(c)
		self.cumul += c
		self.cumul_list.append(self.cumul)



class plotter:
	def __init__(self, hyperparameters, vac_names, test_names):
		self.nc = hyperparameters[0]
		self.T = hyperparameters[1]
		self.xi = hyperparameters[2]
		self.lz = hyperparameters[3]
		self.a = hyperparameters[4]
		self.b = hyperparameters[5]
		self.cc = hyperparameters[6]
		self.dd = hyperparameters[7]
		self.fn = hyperparameters[8]
		self.fp = hyperparameters[9]
		self.p_inf0 = hyperparameters[10]
		self.p_rec0 = hyperparameters[11]
		self.gamma_ = hyperparameters[12]
		self.alpha_ = hyperparameters[13]
		self.N = hyperparameters[14]
		self.bw = hyperparameters[15]
		self.locs = hyperparameters[16]
		self.vac_names = vac_names
		self.test_names = test_names

		self.co = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'indigo', 'lightsteelblue', 'purple', 'teal', 'olive', 'pink', 'honeydew', 'plum', 'darkturquoise', 'navy', 'slategrey', 'aquamarine']


		print('Number of Zones: '+str(self.nc))
		print('Time Horizon: ' + str(self.T))
		print('Total Pop: '+str(np.sum(self.N)))
		print('lower bound: ' +str(-self.p_inf0 * np.sum(self.N)))


	def plot_all(self, costs):
		fig, ax = plt.subplots(1,3)
		legend = []
		counter = 0
		Ntot = np.sum(self.N) * np.ones(self.T)
		for j in range(len(self.test_names)):
			for k in range(len(self.vac_names)):
				cumulative = np.cumsum(costs[j][k], axis=1)
				instant_mean = np.mean(costs[j][k], axis=0)
				cumulative_mean = np.mean(cumulative,axis=0)
				summation_mean = np.cumsum(cumulative_mean)
				ax[0].plot(instant_mean, self.co[counter])
				ax[1].plot(cumulative_mean,self.co[counter])
				ax[2].plot(summation_mean, self.co[counter])
				legend.append([self.vac_names[k] + ' + ' + self.test_names[j]])
				counter+=1
		ax[0].legend(legend)
