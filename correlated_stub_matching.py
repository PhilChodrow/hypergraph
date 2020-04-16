# The purpose of this code is to implement a form of correlated stub-matching. 
# We begin with a set of stubs (specified by a degree sequence), as well as a 
# dimension sequence that specifies the size of the groups that should be 
# aggregated into stubs. We then begin to form sets. To do so, we sample a 
# single node, called the "root node". We then query the root node about its 
# history. Nodes that are present in the history of the root-node are 
# up-weighted for inclusion into the current edge. We repeat this process 
# until all stubs are accounted for and edges constructed. We may optionally 
# remove degeneracies in the resulting graph. This code needs only to 
# construct a list of appropriately-indexed lists, which can then be used 
# as input to the hypergraph class for further analysis. 

import numpy as np
from scipy.sparse import dok_matrix 

class correlated_stub_matcher:

	def __init__(self, D, K, T = None, fun = None):

		'''
		D, np.array(), a degree sequence
		K, np.array(), a dimension sequence
		T, np.array(), a sequence of timestamps

		Must have D.sum() == K.sum()

		fun should take in a stub count vector and a weight vector, returning a normalized probability vector
		'''
		assert D.sum() == K.sum(), "Sum of degrees not equal to sum of edge dimensions!"

		if T is not None: 
			assert np.all(np.diff(T)>=0), "Timestamps are not ordered!"
			assert len(T) == len(K), "Different number of edges and timestamps!"
		else: 
			T = np.arange(len(K))	

		self.D = D
		self.K = K
		self.T = T

		self.n = len(self.D)
		self.m = len(self.K)

		self.M = dok_matrix((self.n, self.m), dtype = int)
		self.stubs = self.D.copy()

		if fun is not None:
			self.weighting_function = fun

	def set_weighting_function(self, fun):
		self.weighting_function = fun

	def clear(self): 
		'''
		Erase current matching (to do a new one)
		'''
		self.M = dok_matrix((self.n, self.m), dtype = int)
		self.stubs = self.D.copy()

	# def construct_edge(self, node_list, check_stubs = False, edge_num = 0):
	# 	'''
	# 	node_list an np.array of ints

	# 	I don't think we should ever have to check_stubs since the stubs should be sampled in weighted fashion from self.stubs to begin with
	# 	'''
	# 	node_list = np.sort(node_list)

	# 	# check if the sampled stubs are present in stub list. 
	# 	# if not, return without doing anything

	# 	if check_stubs:
	# 		for node in node_list:
	# 			if self.stubs[node] < 1:
	# 				return(None)

	# 	self.stubs.subtract(node_list)	# update node list
	# 	self.M[node_list, edge_num] = 1

	# 	return(None)

	def sample_root(self):
		p = self.stubs / self.stubs.sum()
		i = np.random.choice(self.n, p = p)
		return(i)

	def get_history(self, i):
		'''
		i a node
		Get the columns of self.M in which i is nonzero
		'''
		
		# ind = np.where(self.M[i] > 0)[1]
		ind = (self.M[i] > 0).nonzero()[1]
		return(self.M[:,ind]) # matrix of columns of M in which i is nonzero

	def get_weights(self, i, t = None):
		'''
		t the timestep
		i the root node
		eventually we may want a temporal decay kernel for finite memory, not implemented at the moment
		'''
		H = self.get_history(i)
		# to convolve this with a temporal kernel or something
		counts = np.asarray(H.sum(axis = 1)).squeeze() 
		weights = self.weighting_function(self.stubs, counts)
		return(weights)

	def sample_edge(self, edge_num):
		'''
		t the timestep
		'''
		# uniformly random root
		t = self.T[edge_num]
		i = self.sample_root()
		self.M[i,edge_num] += 1
		self.stubs[i] -= 1

		weights = self.get_weights(i, t)

		k = self.K[edge_num]

		for j in range(k-1):
			while True: 
				ell = np.random.choice(np.arange(self.n), p = weights)
				if self.stubs[ell] > 0:
					self.stubs[ell] -= 1
					self.M[ell,edge_num] += 1
					break

	def stub_matching(self):
		for k in range(self.m):
			self.sample_edge(k)

	def get_edge_list(self):
		edge_list = [[]]
		
		nz = self.M.nonzero()

		e = 0
		for i in range(len(nz[1])):
			edge_list[e].append(nz[0][i])
			if i < len(nz[1]) - 1:
				e_next = nz[1][i+1]
				if e_next > e:
					edge_list.append([])
					e += 1

		edge_list = [sorted(e) for e in edge_list]
		return(edge_list)


		

