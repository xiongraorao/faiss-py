# faiss-py

利用restful api来实现faiss的索引操作。

# API文档

## 向量添加

请求地址：http://localhost:2344/add

请求方式：post

请求类型：application/json

**输入参数：**

| 是否必选 | 参数名 | 类型 | 参数说明 |
| :------: | :----: | :--: | :------: |
| 必选 | ntotal | `Int` | 需要添加的向量个数 |
| 必选 | data | `Object` | 向量唯一标识及对应特征向量 |
| 必选 |  data.ids | `Array<Int>` | 特征向量的id列表 |
| 必选 |  data.vectors | `Array<Int>` | 特征向量数组 |

请求示例：

``` json
{
    "ntotal": 3,
    "data": {
        "ids": [1,2,3], 
        "vectors": [[0.12,0.18,...,0.19], 
                  [0.12,0.18,...,0.19], 
                  [0.12,0.18,...,0.19]]
    }
}
```

**返回参数：**

返回类型：JSON

| 参数名 | 类型 | 参数说明 |
| :----: | :--: | :------: |
| time_used | `Int` | 整个请求所花费的时间，单位为毫秒 |
| rtn | `Int` | 返回值，0表示请求成功，非0表示失败 |
| message | `String` | 请求结果描述信息 |

返回示例：

``` json
{"time_used": 99, "rtn": 0, "message": "vectors added successfully"}
```

## 向量查找

请求地址：http://localhost:2344/search

请求方式：post

请求类型：application/json

**输入参数：**

| 是否必选 | 参数名 | 类型 | 参数说明 |
| :------: | :----: | :--: | :------: |
| 必选 | qtotal | `Int` | 需要查找的向量个数 |
| 必选 | topk | `Int` | 需要查找的近邻数 |
| 必选 | queries | `Array<Array<float>>` | 待查找的特征向量列表 |

请求示例：

``` json
{
    "qtotal": 3, 
    "topk":4, 
    "queries":[ [0.12,0.18,...,0.19], 
                [0.12,0.18,...,0.19], 
                [0.12,0.18,...,0.19]]
}
```

**返回参数：**

返回类型：JSON

| 参数名 | 类型 | 参数说明 |
| :----: | :--: | :------: |
| time_used | `Int` | 整个请求所花费的时间，单位为毫秒 |
| rtn | `Int` | 返回值，0表示请求成功，非0表示失败 |
| message | `String` | 请求结果描述信息 |
| rtopk | `Int` | 实际的topk的大小，小于等于输入的topk
| results | `Object` | 查询的结果 |
| results.distances | `Array<Array<float>>` | 当前待查向量的rtopk个最近邻的距离
| results.lables | `Array<Array<int>>` | 当前待查向量的rtopk个最近邻的向量的id

返回示例：

``` json
{
	"time_used": 23, 
	"rtn": 0, 
	"rtopk": 4,
	"results": {
		"distances": [[0.9, 0.8, 0.7,0.6],
						[0.92, 0.82, 0.72,0.62],
						[0.93, 0.83, 0.73,0.63]], 
		"lables": [[1,2,4],
					[5,6,7],
					[9,19,20]]
	},
	"message": "search successfully"
}
```

## 向量删除

请求地址：http://localhost:2344/del

请求方式：post

请求类型：application/json

**输入参数：**

| 是否必选 | 参数名 | 类型 | 参数说明 |
| :------: | :----: | :--: | :------: |
| 必选 | ids | `Array<int>` | 需要删除的向量的id列表 |

请求示例：

``` json
{"ids": [1,2,3]}
```

**返回参数：**

返回类型：JSON

| 参数名 | 类型 | 参数说明 |
| :----: | :--: | :------: |
| time_used | `Int` | 整个请求所花费的时间，单位为毫秒 |
| rtn | `Int` | 返回值，0表示请求成功，非0表示失败 |
| message | `String` | 请求结果描述信息 |

返回示例：

``` json
{"time_used": 0, "rtn": 0, "message": "delete successfully"}
```

## 常见错误代码

| rtn | 说明 |
| :----: | :------: |
| -1 | 输入参数的长度不匹配
| -2 | 输入的向量的维度和系统维度不一致
| -3 | 调用faiss出错
