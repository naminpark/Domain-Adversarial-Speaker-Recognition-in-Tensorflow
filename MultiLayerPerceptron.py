#__author__ = 'naminpark'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from mfcc_feature_extraction import *
import copy


TRpath = './VAD/TRAIN/'
TEpath = './VAD/TEST/'


DR=data_RW()

GMix=2
n_mfcc =20
n_class =15



DR.setVariable(n_mfcc,n_class,GMix)


Trfeat = DR.csvRead(TRpath+"input_feat_TRAIN.csv")
Trlabel= DR.csvRead(TRpath+"input_label_TRAIN.csv")

Tefeat = DR.csvRead(TEpath+"input_feat_TEST.csv")
Telabel= DR.csvRead(TEpath+"input_label_TEST.csv")


DR.setValue(Trfeat,Trlabel)



# Xavier Init
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


# Parameters
beta = 0.01
training_epochs = 1000
batch_size      = 400
display_step    = 10

# Network Parameters
n_input    = n_mfcc*2*GMix # MFCC Feature
n_hidden_1 = 2048 # 1st layer num features
n_hidden_2 = 2048 # 2nd layer num features
n_hidden_3 = 1024 # 3rd layer num features
n_hidden_4 = 512 # 4th layer num features
n_hidden_5 = 512 # 5th layer num features
n_classes  = 15 # speaker classes

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

dropout_keep_prob = tf.placeholder("float")

lr = tf.placeholder("float")

scale1= tf.Variable(tf.ones([n_hidden_1]))
beta1 = tf.Variable(tf.zeros([n_hidden_1]))

scale2= tf.Variable(tf.ones([n_hidden_3]))
beta2 = tf.Variable(tf.zeros([n_hidden_3]))

scale3= tf.Variable(tf.ones([n_hidden_4]))
beta3 = tf.Variable(tf.zeros([n_hidden_4]))

# Create model
def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])

    _mean1, _var1 = tf.nn.moments(layer_1, [0])
    BN1 = tf.nn.batch_normalization(layer_1, _mean1, _var1, beta1,scale1, 0.0001)

    layer_1=tf.nn.relu(BN1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))

    layer_3=tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])

    _mean2, _var2 = tf.nn.moments(layer_3, [0])
    BN2 = tf.nn.batch_normalization(layer_3, _mean2, _var2, beta2,scale2, 0.0001)

    layer_3=tf.nn.relu(BN2)

    layer_4 = tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4'])
    _mean3, _var3 = tf.nn.moments(layer_4, [0])
    BN3 = tf.nn.batch_normalization(layer_4, _mean3, _var3, beta3,scale3, 0.0001)
    layer_4=tf.nn.relu(BN3)

    layer_4 = tf.nn.dropout(tf.nn.relu(layer_4), _keep_prob)

    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))

    return (tf.nn.softmax(tf.matmul(layer_5, _weights['out']) + _biases['out'])) # No need to use softmax??

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("h1", shape=[n_input, n_hidden_1],    initializer=xavier_init(n_input,n_hidden_1)),
    'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2], initializer=xavier_init(n_hidden_1,n_hidden_2)),
    'h3': tf.get_variable("h3", shape=[n_hidden_2, n_hidden_3], initializer=xavier_init(n_hidden_2,n_hidden_3)),
    'h4': tf.get_variable("h4", shape=[n_hidden_3, n_hidden_4], initializer=xavier_init(n_hidden_3,n_hidden_4)),
    'h5': tf.get_variable("h5", shape=[n_hidden_4, n_hidden_5], initializer=xavier_init(n_hidden_4,n_hidden_5)),

    'out': tf.get_variable("out", shape=[n_hidden_5, n_classes], initializer=xavier_init(n_hidden_5,n_classes))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'b5': tf.Variable(tf.zeros([n_hidden_5])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

inf=1e-7
# Define loss and optimizer
#cost = tf.reduce_mean(tf.pow(pred- y,2))
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+inf)))


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+inf))) +beta * tf.nn.l2_loss(weights['h1'])+beta * tf.nn.l2_loss(weights['h2'])
+beta * tf.nn.l2_loss(weights['h3'])+beta * tf.nn.l2_loss(weights['h4'])+beta * tf.nn.l2_loss(weights['out'])
+beta * tf.nn.l2_loss(biases['b1'])+beta * tf.nn.l2_loss(biases['b2'])+beta * tf.nn.l2_loss(biases['b3'])
+beta * tf.nn.l2_loss(biases['b4'])+beta * tf.nn.l2_loss(biases['b5'])+beta * tf.nn.l2_loss(biases['out'])# Softmax loss
optimizer = tf.train.AdamOptimizer(lr).minimize(cost) # Adam Optimizer
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer

# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

print ("Network Ready")

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int((DR.dataX.shape[0])/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = DR.next_batch(i,batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 0.5,lr: 0.0001*0.99**(i)})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob:1.})/total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={x: Tefeat, y: Telabel, dropout_keep_prob:1.})
        print ("Training accuracy: %.3f" % (train_acc))

print ("Optimization Finished!")


test_acc = sess.run(accuracy, feed_dict={x: Tefeat, y: Telabel, dropout_keep_prob:1.})
print ("Training accuracy: %.3f" % (test_acc))


#from sklearn.metrics import roc_curve, auc

#result= sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.})

#fpr, tpr, threshold = roc_curve(Telabel, sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.}), pos_label=1)
#EER = threshold(np.argmin(abs(tpr-fpr)))
