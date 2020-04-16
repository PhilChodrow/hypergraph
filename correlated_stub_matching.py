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
import networkx as nx
from scipy.sparse import dok_matrix 

class correlated_stub_matcher:

	def __init__(D, K, T = None):

	'''
	D, np.array(), a degree sequence
	K, np.array(), a dimension sequence
	T, np.array(), a sequence of timestamps

	Must have D.sum() == K.sum()
	'''
	assert D.sum() == K.sum(), "Sum of degrees not equal to sum of edge dimensions!"

	if T is not None: 
		assert np.diff(T)>=0, "Timestamps are not ordered!"
		assert len(T) == len(K), "Different number of edges and timestamps!"
	else: 
		T = np.arange(len(K))	

	self.D = D
	self.K = K

	self.T = T

	self.M = dok_matrix(n, m, dtype = int)
	self.stubs = self.D.copy()

	def clear(self): 
		self.M = dok_matrix(n, m, dtype = int)
