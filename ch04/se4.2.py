import tensorflow as tf
t = [[2,3,3],[1,5,5]]
t1= tf.expand_dims(t,0)
with tf.Session() as sess:
    print(sess.run(t1))