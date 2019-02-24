'''
An object of class hypergraph is a list of tuples on a specified node set, which can be implicit. 
It has various methods for returning quantities of interest. 
These include: 

    1. Degree sequence of nodes. 
    2. Degree sequence of hyper-edges. 
    3. Induced graph of simplices, with edges weighted according to dimension of incident faces. 
    4. Simplicial complex representation. 

This would also be a convenient class for implementing Metropolis-Hastings. 
'''

import numpy as np
import networkx as nx
from collections import Counter 
from itertools import accumulate
from bisect import bisect
import random
import itertools
from scipy.special import binom
import random

class hypergraph:
    
    def __init__(self, C, n_nodes = None, node_labels = None):
        self.C = [tuple(sorted(f)) for f in C]
        
        self.nodes = list(set([v for f in self.C for v in f]))
        self.n = max(self.nodes) + 1 #assumes first node is 0
#         if n_nodes is None:
#             self.n = len(self.nodes)
#         else: 
#             self.n = n_nodes
        if node_labels is not None:
            self.node_labels = node_labels
        self.m = len(self.C)
             
        # node degrees
        D = np.zeros(self.n)
        for f in self.C:
            for v in f:
                D[v] += 1
        self.D = D
        
        K = np.array([len(f) for f in self.C])
        self.K = K
        
        self.MH_rounds = 0
        self.MH_steps = 0
                
    def node_degrees(self, by_dimension = False):
        if not by_dimension:
            return(self.D)
        else:
            D = np.zeros((len(self.D), max(self.K)))
            for f in self.C:
                for v in f:
                    D[v, len(f)-1] += 1
            return(D)
    def edge_dimensions(self):
        return(self.K)
    
    def induced_graph(self):
        print('not implemented')
    
    def bipartite_graph(self):
        print('not implemented')
    
    def dual_graph(self):
        H = nx.Graph()

        counts = Counter(self.C)
        d = {v : counts[v] for v in counts}

        H.add_nodes_from(d.keys())
        nx.set_node_attributes(H, values = d, name = 'm')
        
        node_list = list(H.nodes())
        n_nodes = len(node_list)
        for u in range(n_nodes):
            for v in range(u+1, n_nodes):
                j = len(set(node_list[u]).intersection(set(node_list[v])))
                if j > 0:
                    H.add_edge(node_list[u],node_list[v], weight = j)
        return(H)
    
    def get_edges(self, node):
        return([f for f in self.C if node in f])
    
    def remove_degeneracy(self, verbose = True):
        m_degenerate = self.check_degeneracy()
        while self.check_degeneracy() > 0:
            for i in range(len(self.C)):
                while is_degenerate(self.C[i]):
                    j = np.random.choice(range(len(self.C)))
                    f1, f2 = self.C[i], self.C[j]
                    self.C[i], self.C[j] = pairwise_reshuffle(f1, f2, True)
        if verbose:
            print(str(m_degenerate) + ' degeneracies removed, ' + str(self.check_degeneracy()) + ' remain.')
    
    def MH(self, n_steps = 1000, sample_every = 5000, sample_fun = None, verbose = True, label = 'edge', n_clash = 0, message = True, **kwargs):
        if (label == 'edge') or (label == 'stub'):
            self.stub_edge_MH(n_steps = n_steps, sample_every = sample_every, sample_fun = sample_fun, verbose = verbose, label = label, message = message,  **kwargs)
        elif label == 'vertex':
            self.vertex_labeled_MH(n_steps = n_steps, sample_every = sample_every, sample_fun = sample_fun, verbose = verbose, n_clash = n_clash, message = message, **kwargs)
        else:
            print('not implemented')
    
    def stub_edge_MH(self, n_steps = 1000, sample_every = 50, sample_fun = None, verbose = True, label = 'edge', message = True,  **kwargs):
        '''
          Not yet correct for stub-matching, but ok for edge I think.   
        '''
        
        C_new = [list(c) for c in self.C]
        m = len(C_new)
        
        proposal = proposal_generator(m)

        def MH_step(label = 'edge'):
            i, j, f1, f2, g1, g2 = proposal(C_new)
            a = acceptance_prob(f1, f2, g1, g2, label = label)
            if np.random.rand() > a:
                return(False)
            else:
                C_new[i] = sorted(g1)
                C_new[j] = sorted(g2)
                return(True)
            
        # main loop
        sample = sample_fun is not None
        if sample:
            v = {}
        
        
        n = 0
        n_rejected = 0
        
        while n < n_steps:
            if MH_step():
                n += 1
                if n % sample_every == 0:
                    if sample:
                        new = sample_fun(self, **kwargs)
                        v.update({n:new})
                        if verbose:
                            print('Current value: ' + str(new))
                    elif verbose:
                            print('Current iteration: ' + str(n) + '. Steps rejected: ' + str(n_rejected))
            else:
                n_rejected += 1
        if message:
            print(str(n) + ' steps taken, ' + str(n_rejected) + ' proposals rejected.')
        self.C = [tuple(sorted(f)) for f in C_new]
        self.MH_steps += n
        self.MH_rounds += 1
        if sample:
            return v
    
    def vertex_labeled_MH(self, n_steps = 10000, sample_every = 500, sample_fun = None, verbose = False, n_clash = 0, message = True, **kwargs):
        '''
        The parameter n_clash controls the number of permitted clashes within each epoch
        '''
        
        rand = np.random.rand
        randint = np.random.randint
        
        sample = sample_fun is not None
        if sample:
            v = {}
        
        k = 0
        done = False
        c = Counter(self.C)
        
        epoch_num = 0
        n_rejected = 0
        
        m = sum(c.values())

        while not done:
            # initialize epoch
            
            l = list(c.elements())
            
            add = []
            remove = []
            
            end_epoch = False
            num_clash = 0
            
            epoch_num += 1
            
            # within each epoch
            
            k_rand = 20000 # generate lots of random numbers at a time
            k_ = 0
            IJ = randint(0, m, k_rand)
            A = rand(k_rand)
            while True:
                if k_ >= k_rand/2.0:
                    IJ = randint(0, m, k_rand)
                    A  = rand(k_rand)
                    k_ = 0
                i,j = (IJ[k_],IJ[k_+1])
                k_ += 2
                
#                 # pick two simplices
                
# #                 i, j, f1, f2, g1, g2 = proposal(l)
                
#                 i,j = np.random.randint(0,m,2)
                f1, f2 = l[i], l[j]
                while f1 == f2:
                    i,j = (IJ[k_],IJ[k_+1])
                    k_ += 2
                    f1, f2 = l[i], l[j]
                if A[k_] > 1.0 /(c[f1] * c[f2]):
                    n_rejected += 1
                else: # if proposal was accepted
                    g1, g2 = pairwise_reshuffle(f1, f2, True)    
                    if (f1 == g1) or (f1 == g2):
                        n_rejected += 1
                    else: # if proposal was not for an identical state
                        
                        # check these simplices against the remove list
                        num_clash += remove.count(f1) + remove.count(f2)
                        if (num_clash >= n_clash) & (n_clash >=1):
                            break
                        else:
                            remove.append(f1)
                            remove.append(f2)
                            add.append(g1)
                            add.append(g2)
                            if k % sample_every == 0:
                                if sample:
                                    new = sample_fun(self, **kwargs)
                                    v.update({k:new})
                            k += 1
                        if n_clash == 0:
                            break
                
            add = Counter(add)
            add.subtract(Counter(remove))
            
            c.update(add) 
            done = k>=n_steps
        if message:
            print(str(epoch_num) + ' epochs completed, ' + str(k) + ' steps taken, ' + str(n_rejected) + ' proposals rejected.')
        self.C = [tuple(sorted(f)) for f in list(c.elements())]
        self.MH_steps += k
        self.MH_rounds += 1
        if sample:
            return(v)
        
    
    def check_degeneracy(self):
        
        return np.sum([is_degenerate(f) for f in self.C])

    def MH_with_stopping(self, memory_length = 5, n_steps = 10**5):
    
        deg = self.node_degrees(by_dimension=True).copy()
        deg_unif = np.outer(deg.mean(axis = 1), deg.mean(axis = 0))
        deg_unif = deg_unif / deg_unif.sum()

        def sq_dist(self):
            deg_sample = self.node_degrees(by_dimension=True)    
            deg_sample = deg_sample / deg_sample.sum()
            return ((deg_sample - deg_unif)**2).mean()

        v = np.array([])
        done = False

        i = 0
        
        while not done:
            self.MH(edge_labeled=True, 
                 joint=False, 
                 n_steps=n_steps, 
                 verbose=False)
            new_val = sq_dist(self)

            if len(v) < memory_length:
                v = np.concatenate(([new_val], v))
            else:
                v = np.concatenate(([new_val], v[0:-1]))
            done = (np.mean(new_val >= v) > .99) & (len(v) == memory_length)
            i += 1
        return(i * n_steps)
        
    def choose_nodes(self, n_samples, choice_function = 'uniform'):
        
        D = self.node_degrees()
        
#         def uniform(x):
#             ind = np.random.choice(x, size = 2, replace = False)
#             return(ind)
        
        def uniform(x):
            i = np.random.randint(len(x))
            j = i
            while i == j:
                j = np.random.randint(len(x))
            return(np.array([x[i],x[j]]))
        
        def top_2(x):
            ind = np.argpartition(D[x,], -2)[-2:]
            y = np.array(x)[ind]
            random.shuffle(y)
            return(y)
        
        def top_bottom(x):
            top = np.argmax(D[x,])
            bottom = np.argmin(D[x,])
            y = np.array(x)[[bottom, top]]
            random.shuffle(y)
            return(y)
        
        choice_functions = {
            'uniform': uniform,
            'top_2' : top_2,
            'top_bottom' : top_bottom, 
            'NA' : uniform
        }
        
        n = 0
        v = []
        while True:
            edge = self.C[np.random.randint(self.m)]
            if len(edge) < 2:
                continue
            x = choice_functions[choice_function](edge)
            v.append(x)
            n+=1
            if n > n_samples:
                break
        return(v)
            
    def assortativity(self, n_samples = 10, choice_function = 'uniform', method = 'pearson'):
        D = self.node_degrees()
        arr = np.array(self.choose_nodes(n_samples, choice_function))
        arr = D[arr]
        
        if method == 'spearman':
            order = np.argsort(arr, axis = 0)
            arr = np.argsort(order, axis = 0)
        elif method == 'pearson':
            arr = arr - 1
            
        return(np.corrcoef(arr.T))[0,1]
        
            
        
            

    
    
    
    
        
is_degenerate = lambda x: len(set(x)) < len(x)



def proposal_generator(m):
    def proposal(edge_list):
        i,j = np.random.randint(0,m,2)
        f1, f2 = edge_list[i], edge_list[j]
        g1, g2 = pairwise_reshuffle(f1, f2, True)
        return(i, j, f1, f2, g1, g2)
    return(proposal)

def acceptance_prob(f1, f2, g1, g2, label = 'stub', counts = None):
    if label == 'stub':
        if (g1 == f1) or (g1 == f2):
            J = len(set(f1).intersection(f2))
            return(1.0 - 2.0**(-J))
    elif label == 'edge':
        if f1 == f2:
            return 0
        elif (g1 == f1) or (g1 == f2):
            J = len(set(f1).intersection(f2))
            return(1 - 2.0**(-(J+1)))
    elif label == 'vertex':
        if (f1 == g1) or (f1 == g2):
            return(0)
        else:
            return(1.0 /(counts[f1] * counts[f2]))
    return(1.0) # if nothing is funky


# def pairwise_reshuffle(f1, f2, preserve_dimensions = True):
#     '''

#     '''
#     f = list(f1) + list(f2)
#     s = set(f)
    
#     intersection = set(f1).intersection(set(f2))
#     ix = list(intersection)
    
#     g1 = ix.copy()
#     g2 = ix.copy()
    
#     for v in ix:
#         f.remove(v)
#         f.remove(v)
    
#     for v in f:
#         if (len(g1) < len(f1)) & (len(g2) < len(f2)):
#             if np.random.rand() < .5:
#                 g1.append(v)
#             else:
#                 g2.append(v)
#         elif len(g1) < len(f1):
#             g1.append(v)
#         elif len(g2) < len(f2):
#             g2.append(v)
#     if len(g1) != len(f1):
#         print('oops')
#         print(f1, f2, g1, g2)
#     return (tuple(sorted(g1)), tuple(sorted(g2)))
  
    
    
    
def pairwise_reshuffle(f1, f2, preserve_dimensions = True):
    
    # easy case: disjoint binary edges
    if (len(f1) == 2) & (len(f2) == 2):
        if len(set(f1).intersection(set(f2))) == 0: 
            return(tuple(sorted([f1[0], f2[1]])),tuple(sorted([f2[0], f1[1]])))
    
    
    
    f1 = set(f1)
    f2 = set(f2)
    ix = f1.intersection(f2)
    
    # slightly less easy case: disjoint intersection
    
#     if len(ix) == 0:
#         diff = list(f1.union(f2))
#         random.shuffle(diff)
#         g1 = diff[0:len(f1)]
#         g2 = diff[len(f1):len(f2)]
#         return(g1, g2)
    
    # worst case: there is overlap
    diff = f1.symmetric_difference(f2)
    diff = list(diff)
    
    len_ix = len(ix)
    i = len(f1)
    j = len(f2)
    
    g1 = ix.copy()
    g2 = ix.copy()
    
    random.shuffle(diff)
        
    g1 = g1.union(set(diff[0:i-len_ix]))
    g2 = g2.union(set(diff[(i-len_ix):len(diff)]))
   
    return(tuple(sorted(g1)), tuple(sorted(g2)))
     
def line_graph(C, weighted = False, as_hyper = False, multi = True):
    '''
    Compute the line graph corresponding to a given hypergraph. Can be slow when many high-dimensional edges are present. 
    '''
    if not as_hyper:
        if multi:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
        G.add_nodes_from(C.nodes)
        for f in C.C:
            if weighted:
                if len(f) >= 2:
                    G.add_edges_from(combinations(f, 2), weight = 1.0/(len(f) - 1))
            else :
                G.add_edges_from(combinations(f, 2))
        return(G)
    else:
        G = [f for F in C.C for f in combinations(F, 2)]
        return(hypergraph(G, n_nodes = len(C.nodes)))
  
def line_graph(C, weighted = False, as_hyper = False, multi = True):
    '''
    Compute the line graph corresponding to a given hypergraph. Can be slow when many high-dimensional edges are present. 
    '''
    if not as_hyper:
        if multi:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
        G.add_nodes_from(C.nodes)
        for f in C.C:
            if weighted:
                if len(f) >= 2:
                    G.add_edges_from(itertools.combinations(f, 2), weight = 1.0/(len(f) - 1))
            else :
                G.add_edges_from(itertools.combinations(f, 2))
        return(G)
    else:
        G = [f for F in C.C for f in itertools.combinations(F, 2)]
        return(hypergraph(G, n_nodes = len(C.nodes)))