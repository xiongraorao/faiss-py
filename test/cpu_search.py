import time

import numpy as np

d = 256                           # dimension
nb = 1000000                   # database size
nq = 1                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 100000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 100000.

start = time.time()
ret = np.dot(xq,xb.T)
np.sort(ret, axis=1)
print(ret)
print(time.time() - start)