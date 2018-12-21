import json
import numpy as np
import requests
d = 256
nb = 1000
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 10000.
xq = xb[:5]

headers = {'content-type': "application/json"}
base_url = 'http://192.168.1.6:2344'

def add():
    url = base_url + '/add'
    data = {'ntotal': nb, 'data': {'ids': list(range(nb)), 'vectors': xb.tolist()}}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

def search():
    url = base_url + '/search'
    data = {'qtotal': 5, 'topk':10, 'queries': xq.tolist()}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

def delete():
    url = base_url + '/del'
    data = {'ids': list(range(1,7))}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(response.json())

def reset():
    url = base_url + '/reset'
    response = requests.get(url)
    print(response.json())

add()
search()
delete()
reset()