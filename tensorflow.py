import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# weights and biases for the first layer
tf.set_random_seed(0)
W_1 = tf.Variable(tf.random_normal([784,30]))
b_1 = tf.Variable(tf.random_normal([30]))
# compute activation for the hidden layer
h1 = tf.nn.sigmoid(tf.matmul(x,W_1)+b_1) # shape = [None,30]

# weights and biases for the second layer
W_2 = tf.Variable(tf.random_normal([30,10]))
b_2 = tf.Variable(tf.random_normal([10]))
# compute activation for the output layer
y = tf.nn.sigmoid(tf.matmul(h1,W_2)+b_2) # shape = [None,10]
# mean square error as the cost function
MSE = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y)))
# specify the method used to train (here, a variant of gradient descend)
train_step = tf.train.AdamOptimizer(1e-4).minimize(MSE)
# compute the accuracy of prediction
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# now that the graph is constructed, it's time to let numbers/data flow through it!
# initialize all variables
session.run(tf.initialize_all_variables())
# set batch size for stochastic gradient descend
batch_size = 100
# perform gradient descend and calculate accuracy after each 10**4 update
for i in range(10**5):
    if i%10**4 == 0:
        print("test accuracy: {}".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
    #for _ in range(len(mnist.train.labels)//batch_size):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})                                                           