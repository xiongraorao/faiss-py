import os
import time
import faiss
import numpy as np
import json

import re
from flask import Flask, request

from util.date import time_to_date
from util.error import *
from util.http import check_param, update_param
from util.logger import Log

app = Flask(__name__)
logger = Log('app')
config = {}
with open('config.json') as f:
    config = json.load(f)
print('======== system config ========')
for key, value in config.items():
    logger.info(key, ":", value)
logger.info('system start time: ', time_to_date(time.time()))
print('======== system config ========')

index = faiss.IndexFlatIP(config['dim'])
index = faiss.IndexIDMap(index)
logger.info('is trained:', index.is_trained)

@app.route('/add', methods=['POST'])
def add():
    start = time.time()
    data = request.data.decode('utf-8')
    necessary_params = {'ntotal', 'data'}
    default_params = {'group_id': 0}
    ret = {'time_used': 0, 'rtn': -1}
    # 检查json 格式
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        logger.warning(GLOBAL_ERR['json_syntax_err'])
        ret['message'] = GLOBAL_ERR['json_syntax_err']
        return json.dumps(ret)
    legal = check_param(set(data), necessary_params, set(default_params))
    if not legal:
        logger.warning(GLOBAL_ERR['param_err'])
        ret['message'] = GLOBAL_ERR['param_err']
        return json.dumps(ret)
    data = update_param(default_params, data)
    ids = np.array(data['data']['ids'], dtype=np.int64)
    vectors = np.array(data['data']['vectors'], dtype=np.float32)
    if len(ids) != len(vectors):
        logger.error('输入特征向量和id的长度不匹配')
        ret['rtn'] = -1
        ret['message'] = SEAERCH_ERR['len_err']
        return json.dumps(ret)
    if len(vectors[0]) != config['dim']:
        logger.error('输入特征向量的维度错误')
        ret['rtn'] = -2
        ret['message'] = SEAERCH_ERR['dim_err']
        return json.dumps(ret)
    try:
        index.add_with_ids(vectors, ids)
        date = time_to_date(time.time())[:10]
        with open(os.path.join(config['index_path'], 'index-%s.log'%date), 'a+') as f:
            for k,v in dict(zip(data['data']['ids'], data['data']['vectors'])).items():
                data = {'id': k, 'vector': v, 'op': 'add'}
                f.write(json.dumps(data)+'\n')
        ret['time_used'] = round((time.time() - start) * 1000)
        ret['message'] = SEAERCH_ERR['add_success']
        ret['rtn'] = 0
    except Exception as e:
        logger.error(e)
        ret['message'] = GLOBAL_ERR['unknow_err']
        ret['rtn'] = -3
    finally:
        logger.info('index total == > %d elements' % index.ntotal)
        return json.dumps(ret)


@app.route('/search', methods=['POST'])
def search():
    start = time.time()
    data = request.data.decode('utf-8')
    necessary_params = {'qtotal', 'topk', 'queries'}
    default_params = {'group_id': 0, 'start': 20180101, 'end': 20500101}
    ret = {'time_used': 0, 'rtn': -1}
    # 检查json 格式
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        logger.warning(GLOBAL_ERR['json_syntax_err'])
        ret['message'] = GLOBAL_ERR['json_syntax_err']
        logger.info('faiss info: ', index.display())
        return json.dumps(ret)
    legal = check_param(set(data), necessary_params, set(default_params))
    if not legal:
        logger.warning(GLOBAL_ERR['param_err'])
        ret['message'] = GLOBAL_ERR['param_err']
        return json.dumps(ret)
    data = update_param(default_params, data)
    if data['qtotal'] != len(data['queries']):
        logger.error('输入待查询的特征向量和qtotal的大小不匹配')
        ret['rtn'] = -1
        ret['message'] = SEAERCH_ERR['len_err']
        return json.dumps(ret)
    queries = np.array(data['queries'], dtype=np.float32)
    k = data['topk']
    try:
        distances, labels = index.search(queries, k)
        ret['results'] = {'distances': distances.tolist(), 'lables': labels.tolist()}
        ret['time_used'] = round((time.time() - start) * 1000)
        ret['rtn'] = 0
        ret['message'] = SEAERCH_ERR['search_success']
    except Exception as e:
        logger.error(e)
        ret['message'] = GLOBAL_ERR['unknow_err']
        ret['rtn'] = -3
    finally:
        logger.info('index total == > %d elements' % index.ntotal)
        return json.dumps(ret)


@app.route('/del', methods=['POST'])
def delete():
    start = time.time()
    data = request.data.decode('utf-8')
    necessary_params = {'ids'}
    default_params = {'group_id': 0, 'timestamps': 0}
    ret = {'time_used': 0, 'rtn': -1}
    # 检查json 格式
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        logger.warning(GLOBAL_ERR['json_syntax_err'])
        ret['message'] = GLOBAL_ERR['json_syntax_err']
        return json.dumps(ret)
    legal = check_param(set(data), necessary_params, set(default_params))
    if not legal:
        logger.warning(GLOBAL_ERR['param_err'])
        ret['message'] = GLOBAL_ERR['param_err']
        return json.dumps(ret)
    data = update_param(default_params, data)
    ids = np.array(data['ids'], dtype=np.int64)
    try:
        index.remove_ids(ids)
        date = time_to_date(time.time())[:10]
        with open(os.path.join(config['index_path'], 'index-%s.log'%date), 'a+') as f:
            for id in data['ids']:
                data = {'id': id, 'op': 'rm'}
                f.write(json.dumps(data)+'\n')
        ret['time_used'] = round((time.time() - start) * 1000)
        ret['message'] = SEAERCH_ERR['delete_success']
        ret['rtn'] = 0
    except Exception as e:
        logger.error(e)
        ret['message'] = GLOBAL_ERR['unknow_err']
        ret['rtn'] = -3
    finally:
        logger.info('index total == > %d elements' % index.ntotal)
        return json.dumps(ret)


@app.route('/reset', methods=['GET'])
def reset():
    start = time.time()
    index.reset()
    ret = {'rtn': 0, 'time_used': round((time.time() - start) * 1000), 'message': SEAERCH_ERR['reset_success']}
    logger.info('index total == > %d elements' % index.ntotal)
    return json.dumps(ret)

def init_index():
    '''
    每个文件索引格式：
    {"id": 123, "vector": [xxx], "op":"add"}
    {"id": 123, "op":"rm"}
    :return:
    '''
    logger.info('==================================')
    logger.info('start initialize index')
    if not os.path.exists(config['index_path']):
        os.mkdir(config['index_path'])
    # load all index
    start = time.time()
    pattern = re.compile(r'index-.*?\.log')
    for index_file in os.listdir(config['index_path']):
        match = pattern.match(index_file)
        add_ids = []
        add_vectors = []
        rm_ids = []
        if match:
            logger.info('load index, file = %s'%match.group())
            for line in open(os.path.join(config['index_path'], match.group()), 'r'):
                data = json.loads(line[:-1])
                if data['op'] == 'add':
                    add_ids.append(data['id'])
                    add_vectors.append(data['vector'])
                elif data['op'] == 'rm':
                    rm_ids.append(data['id'])
            add_ids = np.array(add_ids, dtype=np.int64)
            add_vectors = np.array(add_vectors, dtype=np.float32)
            rm_ids = np.array(rm_ids, dtype=np.int64)
            index.add_with_ids(add_vectors, add_ids)
            index.remove_ids(rm_ids)
            logger.info('index total == > %d elements' % index.ntotal)
    logger.info('index initialize successfully')
    logger.info('index total == > %d elements' % index.ntotal)
    logger.info('index load cost time: %d ms' % round((time.time() - start) * 1000))
    logger.info('==================================')

if __name__ == '__main__':
    # 根据日志重建索引
    init_index()
    app.run(config['host'], config['port'])
