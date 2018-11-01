import tensorflow as tf

########################     样例4-1 session   ###################
# hello = tf.constant("hello,Tensorflow!")
# print(hello) #打印的是类型？
# sess = tf.Session()
# print(sess.run(hello)) #只有运行才能获得hello的返回值
# sess.close()


########################     样例4-2  with   ###################

# a = tf.constant(3)
# b = tf.constant(4)
#
# with tf.Session() as sess:
#     print("相加：%i" %sess.run(a+b))
#     print("相乘：%i" %sess.run(a*b))

########################     样例4-3  注入机制   ###################
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b)
# saver =
savedir = "log/"
with tf.Session() as sess:
    print(sess.run(add,feed_dict={a:3,b:4}))
    print(sess.run(mul,feed_dict={a:3,b:4}))
    print(sess.run([add,mul],feed_dict={a:3,b:4}))
    tf.train.Saver().save(sess,savedir+"add.cpkt")