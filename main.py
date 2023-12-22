import os
import tensorflow as tf
import numpy as np

os.system("rm -rf logs")
os.system("mkdir logs")
tf1 = tf.compat.v1
tf1.disable_eager_execution()
sess = tf1.InteractiveSession()

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf1.placeholder(tf.float32)
prev_avg = tf1.Variable(0.)
update_avg = alpha*curr_value + (1-alpha) * prev_avg
avg_hist = tf1.summary.scalar("running_average", update_avg)
value_hist = tf1.summary.scalar("incoming_values", curr_value)
merged = tf1.summary.merge_all()
writer = tf1.summary.FileWriter("./logs")
init = tf1.global_variables_initializer()

with tf1.Session() as sess:
    sess.run(init)
    #sess._graph._add_op(sess.graph)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg],feed_dict={curr_value:raw_data[i]})
        sess.run(tf1.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)