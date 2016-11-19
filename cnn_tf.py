import tensorflow as tf
import numpy as np
from util import LoadData, DisplayPlot

batch_size = 100


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_bias(shape):
    return tf.Variable(tf.zeros(shape))


def model(X, w1, w2, w3, b1, b2, b3):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1,
                                  strides=[1, 1, 1, 1], padding='SAME') + b1)

    l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,
                                  strides=[1, 1, 1, 1], padding='SAME') + b2)

    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]]
                    )

    pyx = tf.matmul(l3, w3) + b3
    return pyx


inputs_train, inputs_valid, inputs_test, target_train, target_valid, \
    target_test = LoadData('data/toronto_face.npz')

trX = inputs_train.reshape(-1, 48, 48, 1)  # 48x48x1 input img
trY = target_train

valX = inputs_valid.reshape(-1, 48, 48, 1)
valY = target_valid

dataDim = {'h': 48, 'w': 48, 'c': 1}
nfilters = [8, 16]
fsize = 5
num_outpus = 7
eps = 0.001
num_epochs = 40

X = tf.placeholder("float", [None, dataDim['h'], dataDim['w'], dataDim['c']])
Y = tf.placeholder("float", [None, num_outpus])

# 5x5x1 conv, 8 filters
w1 = init_weights([fsize, fsize, dataDim['c'], nfilters[0]])
# 5x5x8 conv, 16 Filters
w2 = init_weights([fsize, fsize, nfilters[0], nfilters[1]])

# FC 1024 inputs, num_outpus  (labels)
w3 = init_weights([1024, num_outpus])

b1 = init_bias([nfilters[0]])
b2 = init_bias([nfilters[1]])
b3 = init_bias([num_outpus])

py_x = model(X, w1, w2, w3, b1, b2, b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdamOptimizer(learning_rate=eps).minimize(cost)

predict_op = tf.argmax(py_x, 1)

# Evaluate model
correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


idx = np.arange(trX.shape[0])

#For plotting the data
train_ce_list = []
train_acc_list = []
valid_ce_list = []
valid_acc_list = []
# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for epoch in range(num_epochs):
        np.random.shuffle(idx)
        trX = trX[idx]
        trY = trY[idx]
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={
                     X: trX[start:end], Y: trY[start:end]})
            cost_tr, acc_tr = sess.run(
                [cost, accuracy], feed_dict={X: trX[start:end], Y: trY[start:end]})
            print "Epoch %i Train CE:%.5f Train Accuracy %.5f" % (epoch, cost_tr, acc_tr)

        # Checking in the val set
        cost_val, acc_val = sess.run(
            [cost, accuracy], feed_dict={X: valX, Y: valY})
        print "Epoch %i Valid CE:%.5f Valid Accuracy %.5f" % (epoch, cost_val, acc_val)
        train_ce_list.append((epoch, cost_tr))
        train_acc_list.append((epoch, acc_tr))
        valid_ce_list.append((epoch, cost_val))
        valid_acc_list.append((epoch, acc_val))
        DisplayPlot(train_ce_list, valid_ce_list, 'Cross Entropy', number=0)
        DisplayPlot(train_acc_list, valid_acc_list, 'Accuracy', number=1)
