#__author__ = 'naminpark'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from mfcc_feature_extraction import *
import copy
from flip_gradient import flip_gradient


TRpath = './VAD/TRAIN/'
TEpath = './VAD/TEST/'


Pred_DR=data_RW()

GMix=2
n_mfcc =20
n_class =15



Pred_DR.setVariable(n_mfcc,n_class,GMix)


Trfeat = Pred_DR.csvRead(TRpath+"input_feat_TRAIN.csv")
Trlabel= Pred_DR.csvRead(TRpath+"input_label_TRAIN.csv")

Tefeat = Pred_DR.csvRead(TEpath+"input_feat_TEST.csv")
Telabel= Pred_DR.csvRead(TEpath+"input_label_TEST.csv")


perm=np.random.permutation(len(Trfeat))




Pred_DR.setValue(Trfeat,Trlabel)



'''
Domain_adversarial data generation

'''

domain_feat = np.vstack([Trfeat, Tefeat])

domain_label = np.vstack([np.tile([1., 0.], [len(Trfeat), 1]),
        np.tile([0., 1.], [len(Tefeat), 1])])


Domain_DR=copy.deepcopy(Pred_DR)



domain_feat,domain_label=Domain_DR.shuffle(domain_feat,domain_label)
Domain_DR.setValue(domain_feat,domain_label)


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
training_epochs = 10000
batch_size      = 100
display_step    = 10

# Network Parameters
n_input    = 40 #n_mfcc*2*GMix # MFCC Feature
fn_hidden_1 = 512 # 1st layer num features
fn_hidden_2 = 512 # 2nd layer num features
fn_hidden_3 = 256 # 3rd layer num features
pn_hidden_4 = 128 # 4th layer num features
pn_hidden_5 = 128 # 5th layer num features
pn_classes  = 15 # speaker classes

dn_hidden_4 = 128 # 4th layer num features
dn_hidden_5 = 64 # 5th layer num features
dn_classes  = 2 # speaker classes

# tf Graph input
x = tf.placeholder("float", [None, n_input])
py = tf.placeholder("float", [None, pn_classes])
dy = tf.placeholder("float", [None, dn_classes])

l = tf.placeholder(tf.float32, [])

dropout_keep_prob = tf.placeholder("float")

lr = tf.placeholder("float")

scale1= tf.Variable(tf.ones([fn_hidden_1]))
beta1 = tf.Variable(tf.zeros([fn_hidden_1]))

scale2= tf.Variable(tf.ones([fn_hidden_3]))
beta2 = tf.Variable(tf.zeros([fn_hidden_3]))

scale3= tf.Variable(tf.ones([pn_hidden_4]))
beta3 = tf.Variable(tf.zeros([pn_hidden_4]))

scale4= tf.Variable(tf.ones([dn_hidden_4]))
beta4 = tf.Variable(tf.zeros([dn_hidden_4]))

# Store layers weight & bias
weights = {
    'fh1': tf.get_variable("fh1", shape=[n_input, fn_hidden_1],    initializer=xavier_init(n_input,fn_hidden_1)),
    'fh2': tf.get_variable("fh2", shape=[fn_hidden_1, fn_hidden_2], initializer=xavier_init(fn_hidden_1,fn_hidden_2)),
    'fh3': tf.get_variable("fh3", shape=[fn_hidden_2, fn_hidden_3], initializer=xavier_init(fn_hidden_2,fn_hidden_3)),

    'ph4': tf.get_variable("ph4", shape=[fn_hidden_3, pn_hidden_4], initializer=xavier_init(fn_hidden_3,pn_hidden_4)),
    'ph5': tf.get_variable("ph5", shape=[pn_hidden_4, pn_hidden_5], initializer=xavier_init(pn_hidden_4,pn_hidden_5)),
    'pout': tf.get_variable("pout", shape=[pn_hidden_5, pn_classes], initializer=xavier_init(pn_hidden_5,pn_classes)),

    'dh4': tf.get_variable("dh4", shape=[fn_hidden_3, dn_hidden_4], initializer=xavier_init(fn_hidden_3,dn_hidden_4)),
    'dh5': tf.get_variable("dh5", shape=[dn_hidden_4, dn_hidden_5], initializer=xavier_init(dn_hidden_4,dn_hidden_5)),
    'dout': tf.get_variable("dout", shape=[dn_hidden_5, dn_classes], initializer=xavier_init(dn_hidden_5,dn_classes))
}
biases = {
    'fb1': tf.Variable(tf.zeros([fn_hidden_1])),
    'fb2': tf.Variable(tf.zeros([fn_hidden_2])),
    'fb3': tf.Variable(tf.zeros([fn_hidden_3])),

    'pb4': tf.Variable(tf.zeros([pn_hidden_4])),
    'pb5': tf.Variable(tf.zeros([pn_hidden_5])),
    'pout': tf.Variable(tf.zeros([pn_classes])),

    'db4': tf.Variable(tf.zeros([dn_hidden_4])),
    'db5': tf.Variable(tf.zeros([dn_hidden_5])),
    'dout': tf.Variable(tf.zeros([dn_classes]))
}

# Model for feature extraction
with tf.variable_scope('feature_extractor'):

    layer_1 = tf.add(tf.matmul(x, weights['fh1']), biases['fb1'])

    _mean1, _var1 = tf.nn.moments(layer_1, [0])
    BN1 = tf.nn.batch_normalization(layer_1, _mean1, _var1, beta1,scale1, 0.0001)

    layer_1=tf.nn.relu(BN1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['fh2']), biases['fb2']))

    layer_3=tf.add(tf.matmul(layer_2, weights['fh3']), biases['fb3'])

    _mean2, _var2 = tf.nn.moments(layer_3, [0])
    BN2 = tf.nn.batch_normalization(layer_3, _mean2, _var2, beta2,scale2, 0.0001)

    feature=tf.nn.relu(BN2)

# MLP for class prediction
with tf.variable_scope('label_predictor'):
    player_4 = tf.add(tf.matmul(feature, weights['ph4']), biases['pb4'])
    _mean3, _var3 = tf.nn.moments(player_4, [0])
    BN3 = tf.nn.batch_normalization(player_4, _mean3, _var3, beta3,scale3, 0.0001)
    player_4=tf.nn.relu(BN3)

    player_4 = tf.nn.dropout(tf.nn.relu(player_4), dropout_keep_prob)

    player_5 = tf.nn.relu(tf.add(tf.matmul(player_4, weights['ph5']), biases['pb5']))

    pred = (tf.matmul(player_5, weights['pout']) + biases['pout']) # No need to use softmax??

    pred_softmax=tf.nn.softmax(tf.matmul(player_5, weights['pout']) + biases['pout'])

    p_loss = tf.nn.softmax_cross_entropy_with_logits( logits=pred, labels=py)


# Small MLP for domain prediction with adversarial loss
with tf.variable_scope('domain_predictor'):

    feat = flip_gradient(feature, l)

    dlayer_4 = tf.add(tf.matmul(feat, weights['dh4']), biases['db4'])
    #_mean4, _var4 = tf.nn.moments(dlayer_4, [0])
    #BN3 = tf.nn.batch_normalization(dlayer_4, _mean4, _var4, beta4,scale4, 0.0001)
    #dlayer_4=tf.nn.relu(BN3)

    #dlayer_4 = tf.nn.dropout(tf.nn.relu(dlayer_4), dropout_keep_prob)

    dlayer_5 = (tf.add(tf.matmul(dlayer_4, weights['dh5']), biases['db5']))

    domain_pred = (tf.matmul(dlayer_5, weights['dout']) + biases['dout']) # No need to use softmax??
    domain_pred_softmax = (tf.nn.softmax(domain_pred)) # No need to use softmax??

    d_loss = tf.nn.softmax_cross_entropy_with_logits( logits=domain_pred, labels=dy)


inf=1e-7
# Define loss and optimizer
#cost = tf.reduce_mean(tf.pow(pred- y,2))
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+inf)))


pred_cost = tf.reduce_mean(p_loss)

domain_cost = tf.reduce_mean(d_loss)

total_loss = tf.add(pred_cost, domain_cost, name='total_loss')

       # Softmax loss
pred_optimizer = tf.train.AdamOptimizer(lr).minimize(pred_cost) # Adam Optimizer
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer
domain_optimizer = tf.train.AdamOptimizer(lr).minimize(domain_cost) # Adam Optimizer

total_optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss) # Adam Optimizer


# Accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(py, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


d_correct_prediction = tf.equal(tf.argmax(domain_pred, 1), tf.argmax(dy, 1))
d_accuracy = tf.reduce_mean(tf.cast(d_correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

print ("Network Ready")

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):

    avg_cost = 0.
    total_batch = int((Pred_DR.dataX.shape[0])/batch_size)


    # Adaptation param and learning rate schedule as described in the paper
    #p = float(epoch) / training_epochs
    #dl = 2. / (1. + np.exp(-10. * p)) - 1
    #learning_rate = 0.01 / (1. + 10 * p)**0.75



    # Loop over all batches
    for i in range(total_batch):

        p = float(i) / training_epochs
        dl = 2. / (1. + np.exp(-10. * p)) - 1
        learning_rate =0.001

        batch_xs, batch_ys = Pred_DR.next_batch(i,batch_size)

        batch_dxs, batch_dys = Domain_DR.next_batch(i,batch_size)

        sess.run(domain_optimizer, feed_dict={x: batch_dxs, dy: batch_dys, l:(4./3.), lr: learning_rate})

        # Fit training using batch data
        sess.run(pred_optimizer, feed_dict={x: batch_xs, py: batch_ys, dropout_keep_prob: 0.7,lr: learning_rate})

        #sess.run(total_optimizer, feed_dict={x: batch_dxs, py: batch_ys, dy: batch_dys,l:dl, dropout_keep_prob: 0.5,lr: learning_rate})


        # Compute average loss
        avg_cost = avg_cost + sess.run(pred_cost, feed_dict={x: batch_xs, py: batch_ys, dropout_keep_prob:1.})/total_batch\
                   + sess.run(domain_cost, feed_dict={x: batch_dxs, dy: batch_dys, l:1.0})/total_batch



    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accuracy, feed_dict={x: Trfeat, py: Trlabel, dropout_keep_prob:1.})
        test_acc = sess.run(accuracy, feed_dict={x: Tefeat, py: Telabel, dropout_keep_prob:1.})
        d_train_acc = sess.run(d_accuracy, feed_dict={x: domain_feat, dy: domain_label, l:1.})
        print ("Training accuracy: %.3f ,  %.3f , %.3f" % (train_acc,test_acc,d_train_acc))

print ("Optimization Finished!")


test_acc = sess.run(accuracy, feed_dict={x: Tefeat, py: Telabel, dropout_keep_prob:1.})
print ("Training accuracy: %.3f" % (test_acc))


#from sklearn.metrics import roc_curve, auc

#result= sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.})

#fpr, tpr, threshold = roc_curve(Telabel, sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.}), pos_label=1)
#EER = threshold(np.argmin(abs(tpr-fpr)))
