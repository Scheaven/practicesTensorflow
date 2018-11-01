#1.9版本之后的tf。不再支持对整数求导！  改成浮点型就可以了
import tensorflow as tf
w1 = tf.Variable([[1.,2]])
w2 = tf.Variable([[3.,4]])

y = tf.matmul(w1, [[9.],[10]])
#grads = tf.gradients(y,[w1,w2])#w2不相干，会报错
grads = tf.gradients(y,[w1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    gradval = sess.run(grads)
    print(gradval)