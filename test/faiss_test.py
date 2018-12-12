import time

import numpy as np
import faiss

d = 256                           # dimension
nb = 1000                   # database size
nq = 100                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 100000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 100000.

index = faiss.IndexFlatIP(d)   # build the index
index = faiss.IndexIDMap(index)
print(index.is_trained)
print('xb shape: ', xb.shape, xb.dtype)
print('id shape: ', np.arange(xb.shape[0]).shape, np.arange(xb.shape[0]).dtype)
index.add_with_ids(xb, np.arange(xb.shape[0]))        # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
print('type D', type(D))
print('type I', type(I))

# start = time.time()
# D, I = index.search(xq, k)     # actual search
# time_used = round( (time.time() - start)* 1000)
# # print(I[:5])                   # neighbors of the 5 first queries
# # print(I[-5:])                  # neighbors of the 5 last queries
# print('time_used', time_used)

start = time.time()
D, I = index.search(xq[:10], k)
time_used = round(time.time() - start)
print('time_used[10]:', time_used)

start = time.time()
D, I = index.search(xq[:100], k)
time_used = round(time.time() - start)
print('time_used[100]:', time_used)

start = time.time()
D, I = index.search(xq[:1000], k)
time_used = round(time.time() - start)
print('time_used[1000]:', time_used)

start = time.time()
D, I = index.search(xq[:10000], k)
time_used = round(time.time() - start)
print('time_used[10000]:', time_used)

start = time.time()
D, I = index.search(xq[:100000], k)
time_used = round(time.time() - start)
print('time_used[100000]:', time_used)

