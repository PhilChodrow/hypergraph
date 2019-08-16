import numpy as np


def adjacency_matrix(G):
    A = np.zeros((G.n, G.n))
    for f in G.C:
        A[f[0], f[1]] += 1
        A[f[1], f[0]] += 1
    return(A)

def degree_sort(A, d):
    row_sorted = A[np.argsort(d)]
    col_sorted = row_sorted[:, np.argsort(d)]
    return(col_sorted)

def W_from_b(b):
    y = 0.5*b.sum()
    BB = np.outer(b, b)
    np.fill_diagonal(BB, 0)
    W = BB / (2*y - BB)
    return(W)

def X_from_b(b):
    y = 0.5*b.sum()
    BB = np.outer(b, b)
    np.fill_diagonal(BB, 0)
    return(BB/(2*y))

def experiment(G, n_stub, n_vertex, n_rounds, sample_after, message_every = 100):
    
    w = np.zeros((G.n, G.n))
    x = np.zeros((G.n, G.n))
    w2 = np.zeros((G.n, G.n))
    
    G.MH(label = 'stub', n_steps = n_stub, verbose = False, message = False)
    print('Completed stub-labeled MCMC with ' + str(int(n_stub)) + ' steps.')
    
    for i in range(n_rounds + sample_after):
        G.MH(label = 'vertex', n_steps = n_vertex, verbose = False, message = False)
        if i % message_every == 0:
            print("Round " + str(i) + " of " + str(n_rounds + sample_after) + ' completed, ' + str(int(i * n_vertex)) + ' steps taken, acceptance rate = ' + str(round(G.acceptance_rate, 4)))
        samples = i - sample_after
        if samples > 0:
            W = adjacency_matrix(G)
            X = 1.0*(W > 0)
            
            w = (samples - 1)/samples*w + 1.0/samples*W
            x = (samples - 1)/samples*x + 1.0/samples*X
            w2 = (samples -1)/samples*w2 + 1.0/samples*(W**2)
    return(w, x, w2)