import tensorflow as tf

tf1 = tf.compat.v1
tf1.disable_eager_execution()
sess = tf1.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spikes = tf1.Variable([False]* (len(raw_data)-1), name="spikes")
sess.run(spikes.initializer)
saver = tf1.train.Saver()
saver.restore(sess, "./spikes.ckpt")
print(spikes.eval())
#for i in range(1, len(raw_data)):
#    if raw_data[i] - raw_data[i-1] > 5:
#        spikes_val = spikes.eval()
#        spikes_val[i-1] = True
#        updater = tf1.assign(spikes, spikes_val).eval()
#save_path = saver.save(sess, "spikes.ckpt")
#print("spikes has been saved successfully to %s" % save_path)
sess.close()
