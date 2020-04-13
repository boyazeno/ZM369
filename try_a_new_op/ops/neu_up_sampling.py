import tensorflow as tf
from tensorflow_core.python.framework import ops
import numpy as np
import sys
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
neu_up_sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'neu_up_sampling_so.so'))

def neu_get_idx_sparse(origin, feature):
    return neu_up_sampling_module.neu_get_idx_sparse(origin, feature)

ops.NoGradient('NeuGetIdxSparse')

def  neu_up_sampling(origin, idx_sparse, feature):

    return neu_up_sampling_module.neu_up_sampling(origin, idx_sparse, feature)

@tf.RegisterGradient('NeuUpSampling')
def _neu_up_sampling_grad(op, grad):
    origin = op.inputs[0]
    idx_sparse = op.inputs[1]
    feature = op.inputs[2]
    return [neu_up_sampling_module.neu_up_sampling_grad(origin,idx_sparse,feature,grad) , None, None]


if __name__=='__main__':
    print(BASE_DIR)
    import numpy as np
    np.random.seed(100)
    origin = np.random.random((2,2,2)).astype('float32')
    feature = np.random.random((2,4,4)).astype('float32')
    with tf.device('/cpu:0'):
        origin_c = tf.constant(origin)
        feature_c = tf.constant(feature)
        idx_sparse = neu_get_idx_sparse(origin_c, feature_c)
        up_sampled = neu_up_sampling(origin_c,idx_sparse,feature_c)
    
    with tf.Session() as sess:
        o, f, u ,i  = sess.run([origin_c, feature_c, up_sampled   ,idx_sparse ])
        print(o)
        print(f)
        print(u)
        print(i)