import tensorflow as tf
import numpy as np
from neu_up_sampling import neu_get_idx_sparse,neu_up_sampling

class UpSamplingTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
      origin = tf.constant(np.random.random((10,16,16)).astype('float32'))
      print(origin)
      feature = tf.constant(np.random.random((10,48,48)).astype('float32'))
      idx_sparse = neu_get_idx_sparse(origin, feature)
      up_sampled = neu_up_sampling(origin,idx_sparse,feature)
      print(up_sampled)
      err = tf.test.compute_gradient_error(origin, (10,16,16), up_sampled, (10,48,48))
      print (err)
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
