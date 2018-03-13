import argparse
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import sys
import struct

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

parser=argparse.ArgumentParser()
parser.add_argument('--lr',help='learning rate',default=0.00001,type=float)
parser.add_argument('--epochs',help='epochs',default=1000,type=int)
parser.add_argument('--batch',help='batch',default=1000,type=int)
parser.add_argument('--logloss', help='logloss model', default='/home/mattliston/ad/LSTM/models/logloss1.proto')
parser.add_argument('--auc', help='auc model', default='/home/mattliston/ad/LSTM/models/auc1.proto')
parser.add_argument('--log', help='name of log file', default='/home/mattliston/ad/LSTM/log/log1.csv')
parser.add_argument('--statesize', help='internal state size', default=100,type=int)
parser.add_argument('--depth', help='network depth', default=1,type=int)
parser.add_argument('--split', help='train/test split', default=0.9,type=int)
args = parser.parse_args()

pos = pd.read_csv('pos_sample.csv').as_matrix()
neg = pd.read_csv('neg_sample.csv')
neg = neg.as_matrix()
neg = neg.astype(int)
neg_test = pd.read_csv('neg_test_sample.csv')
neg_test = neg_test.as_matrix()
data = np.concatenate((pos,neg),axis=0)
np.random.shuffle(data)

train_data = data[0:int(data.shape[0]*args.split),0:6]
train_label = data[0:int(data.shape[0]*args.split),7]
test_data = data[int(data.shape[0]*args.split):data.shape[0],0:6]
test_label = data[int(data.shape[0]*args.split):data.shape[0],7]

for i in range(0,test_label.shape[0]): #ROC requires at least 2 classes
    if test_label[i]==1:
        e=np.zeros((1,data.shape[1]))
        e[0,0:6]=test_data[i]
        e[0,7]=test_label[i]

        dilute=np.zeros((neg_test.shape[0]+1,neg_test.shape[1]),dtype=int)

        print e.shape,dilute.shape
        dilute[0:dilute.shape[0]-1]=neg_test
        dilute[dilute.shape[0]-1]=e
        neg_test=dilute
        break


x = tf.placeholder('float32', [None,6],name='x') ; print x
y = tf.placeholder('float32', [None],name='y') ; print y

layers = [tf.contrib.rnn.LSTMCell(args.statesize) for i in range(0,args.depth)]
output = tf.contrib.rnn.LSTMCell(1)
layers.append(output)
cells = tf.contrib.rnn.MultiRNNCell(layers)
o1,f1 = tf.nn.dynamic_rnn(cells,tf.expand_dims(x,-1),dtype=tf.float32)
n = tf.identity(o1,name='damn_commies')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=tf.reduce_mean(tf.squeeze(n,[2]),axis=1))
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.global_variables_initializer()

best_log=0.693147
best_auc=0
best_negloss=0.693147
best_negauc=0
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,args.epochs):
        larr=[]
        narr=[]
        for j in range(0,1000):
            rand = np.random.randint(train_data.shape[0]-args.batch)
            x_ = train_data[rand:rand+args.batch]
            y_ = train_label[rand:rand+args.batch]
            _,l_,n_ = sess.run([train,loss,norm],feed_dict={x:x_,y:y_})
            larr.append(l_)
            narr.append(n_)

        pred = np.zeros(test_label.shape[0])
        for j in range(0,test_data.shape[0],2000):
            if j==90000:
                p = sess.run('damn_commies:0',feed_dict={x:test_data[j:j+1369]})
                p = sess.run(tf.nn.sigmoid(p))
                pred[j:j+1369] = np.squeeze(p[:,-1,:])
           
            else:
                p = sess.run('damn_commies:0',feed_dict={x:test_data[j:j+2000]})
                p = sess.run(tf.nn.sigmoid(p))
                pred[j:j+2000] = np.squeeze(p[:,-1,:]) #get last output

        n_test = np.zeros(neg_test.shape[0])
        for j in range(0,neg_test.shape[0],2000):
            if j==456000:
                n_p = sess.run('damn_commies:0',feed_dict={x:neg_test[j:j+845,0:6]})
                n_p = sess.run(tf.nn.sigmoid(n_p))
                n_test[j:j+845] = np.squeeze(n_p[:,-1,:])
           
            else:
                n_p = sess.run('damn_commies:0',feed_dict={x:neg_test[j:j+2000,0:6]})
                n_p = sess.run(tf.nn.sigmoid(n_p))
                n_test[j:j+2000] = np.squeeze(n_p[:,-1,:]) #get last output

        with open(args.log,'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([sess.run(tf.losses.log_loss(test_label,pred)),roc_auc_score(test_label,pred),sess.run(tf.losses.log_loss(neg_test[:,7],n_test)),roc_auc_score(neg_test[:,7],n_test)])

        with open('/home/mattliston/ad/LSTM/log/log1_balanced_predictions.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([pred])

        with open('/home/mattliston/ad/LSTM/log/log1_unbalanced_predictions.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([n_test])

        if sess.run(tf.losses.log_loss(test_label,pred))<best_log:
            with open(os.devnull, 'w') as sys.stdout:
                graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),['damn_commies'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', args.logloss, as_text=False)
            best_log = sess.run(tf.losses.log_loss(test_label,pred))

        if roc_auc_score(test_label,pred)>best_auc:
            with open(os.devnull, 'w') as sys.stdout:
                graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),['damn_commies'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', args.auc, as_text=False)
            best_auc = roc_auc_score(test_label,pred)

        if sess.run(tf.losses.log_loss(neg_test[:,7],n_test))<best_negloss:
            with open(os.devnull, 'w') as sys.stdout:
                graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),['damn_commies'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', args.logloss, as_text=False)
            best_negloss = sess.run(tf.losses.log_loss(neg_test[:,7],n_test))

        if roc_auc_score(neg_test[:,7],n_test)>best_auc:
            with open(os.devnull, 'w') as sys.stdout:
                graph=tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),['damn_commies'])
            sys.stdout=sys.__stdout__
            tf.train.write_graph(graph, '.', args.auc, as_text=False)
            best_negauc = roc_auc_score(neg_test[:,7],n_test)


        print 'epoch {:6d} loss {:12.8f} grad {:12.4f} logloss {:12.8f} auc {:12.8f} neglogloss {:12.8f} negauc {:12.8f}'.format(i,np.mean(larr),np.mean(narr),sess.run(tf.losses.log_loss(test_label,pred)),roc_auc_score(test_label,pred),sess.run(tf.losses.log_loss(neg_test[:,7],n_test)),roc_auc_score(neg_test[:,7],n_test))
#        print 'last 50 neg predictions', n_test[-50:] 
    



