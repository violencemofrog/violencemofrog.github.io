# `numpy.newaxis`
本质是None，通过IPython运行`numpy.newaxis`可得：
```python
Type:        NoneType
String form: None
Docstring:   <no docstring>
```

# 通过`numpy.axis`为数组添加维度
```python
In [1]: import numpy

In [2]: x=numpy.arange(3)

In [3]: x
Out[3]: array([0, 1, 2])

In [4]: x.shape
Out[4]: (3,)

In [5]: x[numpy.newaxis,:]
Out[5]: array([[0, 1, 2]])

In [6]: x[numpy.newaxis,:].shape
Out[6]: (1, 3)

In [7]: x[:,numpy.newaxis]
Out[7]:
array([[0],
       [1],
       [2]])

In [8]: x[:,numpy.newaxis].shape
Out[8]: (3, 1)
```
同`numpy.ndarray.reshape`方法，这种增加维度后返回的也是数组的视图而不是副本

该方法相比`numpy.ndarray.reshape`方法略快一点