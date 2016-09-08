import tensorflow as tf
import numpy as np

# load and prepare the mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
mnist_train_labels = np.zeros((len(mnist.train.labels)),dtype=int)
# get the labels in the correct format (int instead of uint8)
for i in range(len(mnist_train_labels)):
    mnist_train_labels[i] = int(mnist.train.labels[i])
mnist_test_labels = np.zeros((len(mnist.test.labels)),dtype=int)
for i in range(len(mnist_test_labels)):
    mnist_test_labels[i] = int(mnist.test.labels[i])
    
# declare the set of columns of the data to be used as features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
# instantiate a deep neural net classifier
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30],
    n_classes=10,
)
# repeatedly fit the classifier to data and evaluate on a test data set
for _ in range(10):
    classifier.fit(x=mnist.train.images, y=mnist_train_labels,steps=100,batch_size=100)
    accuracy_score = classifier.evaluate(x=mnist.test.images, y=mnist_test_labels)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))