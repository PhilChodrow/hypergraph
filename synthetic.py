import numpy as np

def synthetic_hypergraph(n, d_max, k_max):
	D = np.random.randint(1, d_max, n)
	D_ = D.copy()
	K = []
	edge_list = []
	while D_.sum() > 0:
	    k = np.random.randint(1,k_max)
	    ix = np.random.randint(0, n, k)
	    if np.all(D_[ix] > 0):
	        for i in ix:
	            D_[i] -= 1
	        K.append(k)
	        edge_list.append(ix)	
	K = np.array(K)
	return(D, K, edge_list)