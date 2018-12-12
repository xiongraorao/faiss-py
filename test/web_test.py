import json
import numpy as np
import requests
d = 256
nb = 1000
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 10000.
xq = xb[:5]

headers = {'content-type': "application/json"}

def add():
    url = 'http://localhost:2344/add'
    data = {'ntotal': nb, 'data': {'ids': list(range(nb)), 'vectors': xb.tolist()}}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

def search():
    url = 'http://localhost:2344/search'
    data = {'qtotal': 5, 'topk':10, 'queries': xq.tolist()}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

def delete():
    url = 'http://localhost:2344/del'
    data = {'ids': list(range(1,7))}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

add()
search()
delete()