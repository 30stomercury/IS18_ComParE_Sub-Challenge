# selfAssessed
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score
from sklearn.utils import class_weight

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# load data: eGmaps_lld selfAssesed
def load_data(feature_path='Feature/eGemaps_lld/ComParE2018_SelfAssessedAffect/'):
    '''
    shape: [num_data, sequence_length, dimension]
    '''
    #X_train = joblib.load(feature_path+'X_train.pkl')
    #X_devel = joblib.load(feature_path+'X_devel.pkl')
    X_test = np.load(feature_path+'X_test.npy')
    X_train = np.load(feature_path+'X_train.npy')
    X_devel = np.load(feature_path+'X_devel.npy')
    # define label
    df = pd.read_csv(feature_path+'ComParE2018_SelfAssessedAffect.csv')
    y_train = df['Valence'][:846].values
    y_devel = df['Valence'][846:].values
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_devel = le.transform(y_devel)
    y_train = y_train.reshape(len(y_train), 1)
    y_devel = y_devel.reshape(len(y_devel), 1)
    # convert into one-hot label
    enc = preprocessing.OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train)
    y_devel = enc.transform(y_devel)
    return X_train, X_devel, y_train, y_devel

class BatchGenerator:
    def __init__(self, X, y, hparams):
        self.hps = hparams
        D = self.hps.input_dim
        long_ = self.hps.seq_length
        batch_size = self.hps.BATCH_SIZE 
        n = len(X)
        self.batch_xs, self.batch_ys = [], []
        for i in range(n//batch_size+1):
            if i != (n//batch_size):
                batch_x = np.zeros((batch_size, long_, D))
                batch_y = np.zeros((batch_size, 3))
                for j in range(batch_size):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]
                        #print(k)
                    for k in range(k, long_):
                        batch_x[j][k] = np.zeros(D) # padding with 0
                    batch_y[j] = y[i*batch_size+j] # 1-hot vector  
            else:
                batch_x = np.zeros((len(X) % batch_size, long_, D))
                batch_y = np.zeros((len(y) % batch_size, 3))                         
                for j in range((len(y) % batch_size)):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]
                    for k in range(k, long_):
                        batch_x[j][k] = np.zeros(D) # padding with 0      
                
                    batch_y[j] = y[i*batch_size+j] # 1-hot vector 
            self.batch_xs.append(batch_x)
            self.batch_ys.append(batch_y)
        
    def get(self, batch_id):
        return self.batch_xs[batch_id], self.batch_ys[batch_id]

class Bi_RNN:
    def __init__(self, mode, hparams):
        self.mode = mode
        self.hps = hparams

        with tf.variable_scope('rnn_input'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.hps.input_dim])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, 3])

        with tf.variable_scope('rnn_cell'):
            self.cell_units = self.hps.cell_unit #32
            # Forward direction cell
            self.fw_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_units)
            # Backward direction cell
            self.bw_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_units)
            # when training, add dropout to regularize.
            if self.mode == 'train':
                self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell,
                                                            input_keep_prob=self.hps.keep_proba,
                                                            output_keep_prob=self.hps.keep_proba)
                self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell,
                                                            input_keep_prob=self.hps.keep_proba,
                                                            output_keep_prob=self.hps.keep_proba)
            
        with tf.variable_scope('rnn_bidirectional_with_attention_layer'):
            # use dynamic_rnn for different length
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                             cell_bw=self.bw_cell,
                                                             inputs= self.inputs,
                                                             dtype=tf.float32,
                                                             time_major=False) 
            # Attention layer
            outputs = tf.concat(outputs, 2)
            # parameters
            hidden_size = outputs.shape[2].value  # hidden size of the RNN layer
            attention_size = 1
            # Trainable parameters
            W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            v = tf.sigmoid(tf.tensordot(outputs, W, axes=1) + b) 
            vu = tf.tensordot(v, u, axes=1)   # (Batch size,T)
            alphas = tf.nn.softmax(vu)        # (Batch size,T)
            
            W2 = tf.Variable(tf.random_normal([self.hps.seq_length, self.hps.seq_length], stddev=0.1))
            b2 = tf.Variable(tf.random_normal([self.hps.seq_length], stddev=0.1))
            alphas2 = tf.nn.sigmoid(tf.tensordot(alphas, W2, axes=1) + b2)
            alphas2 = tf.nn.softmax(alphas2)
            if self.mode == 'train':
                alphas2 = tf.nn.dropout(alphas2, keep_prob=self.hps.keep_proba)
            
            # Output of Bi-RNN is reduced with attention vector: (Batch size, hidden_size)
            self.outputs = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)

            # fully layer
            full_weight = tf.get_variable('full_weight', shape=[2*self.cell_units, 3], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            full_bias = tf.get_variable('full_bias', shape=[3], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            dense = tf.matmul(self.outputs, full_weight) + full_bias
            dense = tf.nn.relu(dense)
            dense = bn(dense)
            # when training, add dropout to regularize.
            if self.mode == 'train':
                dense = tf.nn.dropout(dense, keep_prob=self.hps.keep_proba)
            
            self.logits = tf.nn.softmax(dense)

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.groundtruths, logits=self.logits)
            # apply gradient clipping
            grad_clip = 1
            var_trainable_op = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, var_trainable_op), grad_clip)
            op = tf.train.AdamOptimizer(self.hps.lr) 
            self.optimizer = op.apply_gradients(zip(grads, var_trainable_op))
            
        with tf.variable_scope('rnn_accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.groundtruths, axis=1),
                predictions=tf.argmax(self.logits, axis=1))
            
        with tf.variable_scope('rnn_uar'):  
            lab_argmax = tf.argmax(self.groundtruths, axis=1)
            pred_argmax = tf.argmax(self.logits, axis=1)
            self.lab_argmax = lab_argmax
            self.pred_argmax = pred_argmax

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.sess.run(tf.local_variables_initializer())   # don't forget to initialise the local variables hidden in the tf.metrics.recall method.
        self.saver = tf.train.Saver()                     # a saver is for saving or restoring your trained weight

    def train(self, batch_x, batch_y):
        #feed dict
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, _ = self.sess.run(
            [self.loss, self.accuracy, self.optimizer], fd)
        lab_argmax= self.sess.run(self.lab_argmax, {self.groundtruths: batch_y})
        pred_argmax= self.sess.run(self.pred_argmax, fd)
        uar = recall_score(lab_argmax, pred_argmax, average='macro')

        return loss, accuracy, uar

    def test(self, batch_x, batch_y):
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y        
        prediction, accuracy, pred_argmax = self.sess.run([self.logits, self.accuracy, self.pred_argmax], fd)
        lab_argmax= self.sess.run(self.lab_argmax, {self.groundtruths: batch_y})
        loss = self.sess.run(self.loss, fd)
        uar = recall_score(lab_argmax, pred_argmax, average='macro')
        logits = self.sess.run(self.logits, fd)

        return loss, accuracy, uar, logits

    def save(self, e):
        self.saver.save(self.sess, self.hps.save_path+'/rnn_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, self.hps.save_path+'/rnn_%d.ckpt' % (e))

# batch norm
def bn(X, eps=1e-8, offset = 0, scale = 1):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        var = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        var = tf.reduce_mean(tf.square(X-mean), 0)
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    else:
        raise NotImplementedError
    return output

# hyperparameter of our network
def get_hparams():
    hparams = tf.contrib.training.HParams(
        EPOCHS=250,
        BATCH_SIZE=128*2,
        input_dim=23,
        seq_length=796,
        lr=0.0005,
        cell_unit=32,
        keep_proba=0.8,
        save_path='./model/brnn')
    return hparams

if __name__ == '__main__':
    # hyperparameters
    hparams = get_hparams()
    # Data generator
    X_train, X_devel, y_train, y_devel = load_data()
    train_batch = BatchGenerator(X_train, y_train, hparams)
    devel_batch = BatchGenerator(X_devel, y_devel, hparams)
    n_train = len(X_train) // hparams.BATCH_SIZE +1
    n_devel = len(X_devel) // hparams.BATCH_SIZE +1
    # model
    model = Bi_RNN(mode='train', hparams=hparams)
    # training
    rec_loss = []
    devel_accuracy = []
    train_uar = []
    devel_uar = []
    EPOCHS = hparams.EPOCHS
    for _epoch in range(EPOCHS):  # train for several epochs
        loss_train = 0
        accuracy_train = 0
        UAR_train = 0
        
        model.mode = 'train'
        for b in range(n_train):  # feed batches one by one
            batch_x, batch_y = train_batch.get(b)
            loss_batch, accuracy_batch, UAR_batch = model.train(batch_x, batch_y)

            loss_train += loss_batch
            accuracy_train += accuracy_batch
            UAR_train += UAR_batch

        loss_train /= n_train
        accuracy_train /= n_train
        UAR_train /= n_train

        model.save(_epoch)  # save your model after each epoch
        rec_loss.append([loss_train, accuracy_train])
        train_uar.append(UAR_train)
        print("Epoch: [%2d/%2d], rnn_loss: %.3f"  % (_epoch+1, EPOCHS, loss_train))
        # validation
        if (_epoch + 1) % 1 == 0:
            accuracy_devel = 0
            UAR_devel = 0
            model.mode = 'test'
            for b in range(n_devel):
                batch_x, batch_y = devel_batch.get(b)
                _, accuracy_batch, UAR_batch, _ = model.test(batch_x, batch_y)
                
                accuracy_devel += accuracy_batch
                UAR_devel += UAR_batch

            accuracy_devel /= n_devel
            UAR_devel /= n_devel
            devel_accuracy.append(accuracy_devel)
            devel_uar.append(UAR_devel)
            print("Accuracy train: %.3f, Accuracy devel: %.3f, UAR_train: %.3f, UAR_devel: %.3f"  % (accuracy_train, 
                                                                                                     accuracy_devel, UAR_train, UAR_devel))
