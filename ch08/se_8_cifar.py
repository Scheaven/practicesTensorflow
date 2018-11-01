# -*- coding: utf-8 -*-
from practicesTensorflow.ch08.cifar10 import cifar10_input
import tensorflow as tf
import pylab

#取数据
batch_size = 128
data_dir = 'cifar10_data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n",image_batch[0])

print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()


# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     image_batch,labels_batch = sess.run([images_test,labels_test])
#
#     print("__\n",image_batch[0])
#     print("__\n",labels_batch[0])
#
#     pylab.imshow(image_batch[0])
#     pylab.show()
#     coord.request_stop()