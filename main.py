import tensorflow as tf

tf1 = tf.compat.v1
tf1.disable_eager_execution()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spikes = tf1.Variable([False]* (len(raw_data)-1), name="spikes")