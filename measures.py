import numpy as np
import pandas as pd
from collections import Counter

def sample_intersection(C, n_samples):
    
    m = len(C.C)
    
    k_rand = 20000
    IJ = np.random.randint(0, m, size = (k_rand+1, 2))
    k_ = 0
    i = 0
    
    counts = Counter()
    
    while i < n_samples:
#         m1, m2 = np.random.randint(0, m, size = 2)
        m1, m2 = IJ[k_,0], IJ[k_,1]
        if m1 == m2:
            pass
        else:
            f1 = C.C[m1]
            f2 = C.C[m2]
            counts[(len(f1), len(f2), len(set(f1) & set(f2)))] += 1
            i += 1
        k_ += 1
        if k_ >= k_rand:
            IJ = np.random.randint(m, size = (k_rand, 2))
            k_ = 0
            
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df['k_1'] = df['index'].apply(lambda x: x[0])
    df['k_2'] = df['index'].apply(lambda x: x[1])
    df['j'] = df['index'].apply(lambda x: x[2])
    df = df.rename({0 : 'n'}, axis = 1)
    df = df.drop('index', axis = 1)
    
    return df