
import numpy as np
import time
import theano
import theano.tensor as T
import gnumpy as gp
from theano import sandbox, Out


a = T.matrix('a')
b = T.matrix('b')
c = T.dot(a,b)
f = theano.function([a,b],Out(sandbox.cuda.basic_ops.gpu_from_host(c),borrow=True))

mat1 = np.asarray(np.random.rand(2000,2000),dtype=np.float32)
mat2 = np.asarray(np.random.rand(2000,2000),dtype=np.float32)
start = time.time()
for i in xrange(20):
    c = f(mat1,mat2)
end = time.time()
print "(theano) Time elapsed : %f second " % (end-start)

a = gp.rand(2000,2000)
b = gp.rand(2000,2000)
start = time.time()
for i in xrange(20):
    c = gp.dot(a, b)
end = time.time()
print "(gnumpy) Time elapsed : %f second " % (end-start)

