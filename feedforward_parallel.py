import tensorflow as tf
import pandas as pd
import csv
import numpy as np

from checkmate import BestCheckpointSaver
import checkmate

input_path = 'emnlp2018.sql.txt'
output_path = 'emnlp2018.sql.csv'
header = ['cnty', '01hea_aar', 'ls09_10_avg', 'hsgradHC03_VC93ACS3yr$10', 'bachdegHC03_VC94ACS3yr$10',
          'logincomeHC01_VC85ACS3yr$10', \
          'unemployAve_BLSLAUS$0910', 'femalePOP165210D$10', 'hispanicPOP405210D$10', 'blackPOP255210D$10',
          'forgnbornHC03_VC134ACS3yr$10', \
          'county_density', 'marriedaveHC03_AC3yr$10', 'median_age']


def readfile(input_path):
    sql_file = open(input_path).readlines()
    table = []
    for line in sql_file:
        if line.startswith('INSERT'):
            table = line.split()[-1]
    return table


def write_csv(mylist, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)

        for row in mylist:
            writer.writerow(row.replace('(', '').split(','))


def init_weights(shape,trainable=True):
    """ Weight initialization """
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.5, mean=0.5),trainable=trainable)
    b = tf.Variable(tf.zeros(1, dtype=tf.float32),trainable=trainable)
    return weights, b


class ffNN():
    tf.reset_default_graph()

    def __init__(self, hidden_nodes=[64], epochs=3, learning_rate=0.1, saveFrequency=1,
                 save_path='./models/ControlOnly', hidden_layers=1, decay=False, decay_step=10, decay_factor=0.7,
                 stop_loss=0.0001, regularization_factor=0.01, keep_probability=0.7, minimum_cost=0.2,
                 activation_function='sigmoid', batch_size=1,shuffle=True,optimizer='Adam',stopping_iteration=10):
        self.hidden_nodes = hidden_nodes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.saveFrequency = saveFrequency
        self.save_path = save_path
        self.hidden_layers = hidden_layers
        self.decay = decay
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        self.stop_loss = stop_loss
        self.keep_probability = keep_probability
        self.regularization_factor = regularization_factor
        self.minimum_cost = minimum_cost
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.optimizer= optimizer
        self.trainable1= True
        self.trainable2= True  
        self.trainable3= True
        self.stopping_iteration = stopping_iteration
        print('model started working :D')

    def forwardprop(self, X, w_in, b_in, keep_prob):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        z = tf.matmul(X, w_in)
        is_train = False

        def retTrue():
            return True

        def retFalse():
            return False

        is_train = tf.cond(tf.less(keep_prob, tf.constant(1.)), retTrue, retFalse)
        z_norm = tf.add(z, b_in)
        # z_norm = tf.layers.batch_normalization(z, training=is_train)
        if self.activation_function == 'relu':
            h = tf.nn.relu(z_norm)
        elif self.activation_function == 'sigmoid':
            h = tf.nn.sigmoid(z_norm)
        elif self.activation_function == 'tanh':
            h = tf.nn.tanh(z_norm)
        else:
            h = z_norm
        # h = tf.nn.sigmoid(tf.matmul(X, w_in))  # The \sigma function
        # h = tf.matmul(X, w_1)
        # h = tf.add(h, b_in)
        # if self.keep_prob <> 1:
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        return h
        # yhat = tf.add(tf.matmul(h, w_out),b_out)

        # if self.hidden_layers >= 2:
        #     h2 = tf.nn.sigmoid(tf.matmul(h,w_hid))
        #    h2 = tf.add(h2,b_hid)
        #   h2 = tf.nn.dropout(h2,keep_prob = keep_prob)
        #  yhat = tf.add(tf.matmul(h2, w_out),b_out)  # The \varphi function

        # if self.hidden_layers ==3:
        #      h3 = tf.nn.sigmoid(tf.matmul(h2,w_hid2))
        #      h3 = tf.add(h3,b_hid2)
        #      h3 = tf.nn.dropout(h3,keep_prob = keep_prob)
        #      yhat = tf.add(tf.matmul(h3, w_out),b_out)  # The \varphi function
        # yhat = tf.nn.dropout(tf.nn.relu(yhat),keep_prob = keep_prob)
        # return yhat

    def initialize(self, x1_size=11,x2_size=11, y_size=1):

        print('initializing parameters')
        self.X1 = tf.placeholder("float", shape=[None, x1_size], name='X1Value')
        self.X2 = tf.placeholder("float", shape=[None, x2_size], name='X2Value')

        self.learning_rate1 = tf.placeholder("float", shape=(), name='learning_rate1')
        self.learning_rate2 = tf.placeholder("float", shape=(), name='learning_rate2')
        self.learning_rate3 = tf.placeholder("float", shape=(), name='learning_rate3')

        #self.learning_rate1 = tf.placeholder("float", shape=(), name='learning_rate1')
        #self.learning_rate2 = tf.placeholder("float", shape=(), name='learning_rate2')
        #self.learning_rate3 = tf.placeholder("float", shape=(), name='learning_rate3')


        self.y = tf.placeholder("float", shape=[None, y_size], name='yValue')
        self.keep_prob = tf.placeholder("float", shape=(), name='keep_prob')
        self.global_step = tf.Variable(0,trainable = False)
        self.initialize_global_step = tf.assign(self.global_step,tf.constant(0))
        # print('self.X size ', self.X.shape)

        # Weight initializations
        # w_in,b_in = init_weights((x_size, self.hidden_nodes))
        # w_hid,b_hid = init_weights((self.hidden_nodes,self.hidden_nodes))
        # w_hid2,b_hid2 = init_weights((self.hidden_nodes,self.hidden_nodes))
        # w_out,b_out = init_weights((self.hidden_nodes, y_size))
        l2_norm1 = 0.
        l2_norm2 = 0.
        l2_norm3 = 0.
        with tf.variable_scope("initialization", reuse=tf.AUTO_REUSE):
            # self.reg_factor = tf.get_variable( name='reg_factor',shape=(), dtype = tf.float32)
            self.reg_factor = self.regularization_factor
            self.increment_global_step = tf.assign_add(self.global_step,1,name = 'increment_global_step')


            w_in1, b_in1 = init_weights((x1_size, self.hidden_nodes[0]),trainable=self.trainable1)
            h_out1 = self.forwardprop(self.X1, w_in1, b_in1, self.keep_prob)
            l2_norm1 = tf.add(tf.nn.l2_loss(w_in1), l2_norm1)
            l2_norm1 = tf.add(tf.nn.l2_loss(b_in1), l2_norm1)
            i = -1
            # Forward propagation
            for i in range(self.hidden_layers - 1):
                w_in1, b_in1 = init_weights((self.hidden_nodes[i], self.hidden_nodes[i + 1]),trainable=self.trainable1)
                h_out1 = self.forwardprop(h_out1, w_in1, b_in1, self.keep_prob)
                l2_norm1 = tf.add(tf.nn.l2_loss(w_in1), l2_norm1)
                l2_norm1 = tf.add(tf.nn.l2_loss(b_in1), l2_norm1)
            w_out1, b_out1 = init_weights((self.hidden_nodes[i + 1], y_size),trainable=self.trainable1)
            l2_norm1 = tf.add(tf.nn.l2_loss(w_out1), l2_norm1)
            l2_norm1 = tf.add(tf.nn.l2_loss(b_out1), l2_norm1)
            self.yhat1 = tf.add(tf.matmul(h_out1, w_out1), b_out1)
            # Backward propagation
            self.loss1 = tf.losses.mean_squared_error(labels=self.y, predictions=self.yhat1)
            #self.loss1 = tf.add(self.loss1, self.reg_factor * l2_norm1)
            if self.decay:
                 learning_rate1 = tf.train.exponential_decay(self.learning_rate1,self.global_step, self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
                 learning_rate1 = self.learning_rate1
            #self.updates1 = tf.train.AdamOptimizer(learning_rate=learning_rate1, beta1=0.99, beta2=0.999).minimize(self.loss1)
            self.updates1 = tf.train.AdamOptimizer(learning_rate=learning_rate1).minimize(self.loss1)
            


            w_in2, b_in2 = init_weights((x2_size, self.hidden_nodes[0]),trainable=self.trainable2)
            h_out2 = self.forwardprop(self.X2, w_in2, b_in2, self.keep_prob)
            l2_norm2 = tf.add(tf.nn.l2_loss(w_in2), l2_norm2)
            l2_norm2 = tf.add(tf.nn.l2_loss(b_in2), l2_norm2)


            i = -1
            # # Forward propagation
            for i in range(self.hidden_layers - 1):
                w_in2, b_in2 = init_weights((self.hidden_nodes[i], self.hidden_nodes[i + 1]),trainable=self.trainable2)
                h_out2 = self.forwardprop(h_out2, w_in2, b_in2, self.keep_prob)
                l2_norm2 = tf.add(tf.nn.l2_loss(w_in2), l2_norm2)
                l2_norm2 = tf.add(tf.nn.l2_loss(b_in2), l2_norm2)
            w_out2, b_out2 = init_weights((self.hidden_nodes[i + 1], y_size),trainable=self.trainable2)
            l2_norm2 = tf.add(tf.nn.l2_loss(w_out2), l2_norm2)
            l2_norm2 = tf.add(tf.nn.l2_loss(b_out2), l2_norm2)
            self.yhat2 = tf.add(tf.matmul(h_out2, w_out2), b_out2)
            # Backward propagation
            self.loss2 = tf.losses.mean_squared_error(labels=self.y, predictions=self.yhat2)
            self.loss2 = tf.add(self.loss2, self.reg_factor * l2_norm2)
            if self.decay:
                learning_rate2 = tf.train.exponential_decay(self.learning_rate2,self.global_step,  self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
                learning_rate2= self.learning_rate2
            self.updates2 = tf.train.AdamOptimizer(learning_rate=learning_rate2).minimize(self.loss2)

            


            w_out3, b_out3 = init_weights((self.hidden_nodes[i + 1]*2, y_size),trainable=self.trainable3)
            self.yhat3 = self.forwardprop(tf.concat([h_out1,h_out2],axis=1), w_out3, b_out3, self.keep_prob)
            #w_out3, b_out3 = init_weights((2, y_size),trainable=self.trainable3)
            #self.yhat3 = self.forwardprop(tf.concat([self.yhat1, self.yhat2],axis=1), w_out3, b_out3, self.keep_prob)
            l2_norm3 = tf.add(tf.nn.l2_loss(w_out3), l2_norm3)
            l2_norm3 = tf.add(tf.nn.l2_loss(b_out3), l2_norm3)
            # self.yhat3 = tf.add(tf.matmul(h_out2, w_out3), b_out3)
            # Backward propagation
            self.loss3 = tf.losses.mean_squared_error(labels=self.y, predictions=self.yhat3)
            self.loss3 = tf.add(self.loss3, self.reg_factor * l2_norm3)
            if self.decay:
               learning_rate3 = tf.train.exponential_decay(self.learning_rate3,self.global_step,  self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
               learning_rate3 = self.learning_rate3
            self.updates3 = tf.train.AdamOptimizer(learning_rate=learning_rate3).minimize(self.loss3)


            self.final_opt = tf.group(self.updates1,self.updates2,self.updates3)
            

    def train(self, train_X, train_y):
            print('x_1 size %s %s' % train_X[0].shape)
            # print('y_2 size %s %s' % train_y.shape)
            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            bestLoss = np.inf
            import os
            import shutil
            best_save_dir = self.save_path + '/bestModel/'
            if os.path.exists(best_save_dir):
                shutil.rmtree(best_save_dir)
            os.makedirs(best_save_dir)
            best_ckpt_saver = BestCheckpointSaver(
                save_dir=best_save_dir,
                num_to_keep=3,
                maximize=False
            )
            train_X1 = train_X[1]
            train_X2 = train_X[0]

            pass_best_epochs = 0
            best_saved_epoch = -10
            phase = 0
            temp_regularization_factor = self.regularization_factor
            temp_keep_prob = self.keep_probability
            max_phase = 3
            for epoch in range(self.epochs):
                # Train with each example
                allcosts = 0
                if phase == 0:
                   learning_rate_phase1 = self.learning_rate 
                   learning_rate_phase2 = 0.
                   learning_rate_phase3 = 0.
                   self.regularization_factor = 0.
                   self.keep_probability = 1.0
                  
                elif phase == 1:
                   learning_rate_phase1 = 0.
                   learning_rate_phase2 = self.learning_rate
                   learning_rate_phase3 = 0.
                   self.regularization_factor = temp_regularization_factor
                   self.keep_probability = temp_keep_prob
                   sess.run(self.initialize_global_step)
                elif phase == 2:
                   learning_rate_phase1 = 0. #float(self.learning_rate)/100
                   learning_rate_phase2 = 0. #float(self.learning_rate)/10
                   learning_rate_phase3 = self.learning_rate
                   self.trainable1= False
                   self.trainable2= False
                   sess.run(self.initialize_global_step)
                else:
                   learning_rate_phase1 = 0. #float(self.learning_rate)/100
                   learning_rate_phase2 = 0. #float(self.learning_rate)/10
                   #learning_rate_phase3 = self.learning_rate
                   self.trainable1= True
                   self.trainable2= True
                   sess.run(self.initialize_global_step)
                import random
                if self.shuffle:
                   s = np.arange(0, len(train_X1), 1)
                   np.random.shuffle(s)
                   train_X1 = train_X1[s]
                   train_X2 = train_X2[s]
                   train_y = train_y[s]
                # print(train_X)
                
                for i in range(0, len(train_X1) - self.batch_size + 1, self.batch_size):
                    # x_input = list(map(float,train_X[i: i + 2].values.tolist()[0]))
                    # x_input = np.array([x_input],dtype=np.float64)
                    x1_input = np.array(train_X1[i: i + self.batch_size], dtype=np.float64)
                    x2_input = np.array(train_X2[i: i + self.batch_size], dtype=np.float64)
                    #y_input = list(map(float,train_y[i: i + 1].values.tolist()[0]))
                    #y_input = np.array([y_input,y_input],dtype=np.float64)
                    y_input = np.transpose(np.array([train_y[i: i + self.batch_size]],dtype=np.float64))
                    # _, myloss, y_out = sess.run([self.updates, self.loss, self.yhat],
                    #                             feed_dict={self.X: x_input, self.y: y_input,
                    #                                        self.keep_prob: self.keep_probability})

                    #batch += 1
                    #sess.run(self.increment_global_step)
                    loss = sess.run([self.final_opt,self.loss1,self.loss2,self.loss3],feed_dict={self.X1: x1_input, self.X2: x2_input, self.y: y_input,
                                                           self.keep_prob: self.keep_probability,self.learning_rate1: self.learning_rate,
                                                              self.learning_rate2: learning_rate_phase2, self.learning_rate3:learning_rate_phase3 })
                    #loss = sess.run([self.final_opt,self.loss1,self.loss2,self.loss3],feed_dict={self.X1: x1_input, self.X2: x2_input, self.y: y_input,
                    #                                     #  self.keep_prob: self.keep_probability,self.learning_rate1: learning_rate_phase1,
                    #                                     #     self.learning_rate2: learning_rate_phase2, self.learning_rate3:learning_rate_phase3 })
                    
                    #allcosts += loss[1]
                    allcosts += loss[min(phase,2)+1]
                # if self.decay is True and epoch % self.decay_step ==0:
                #    self.learning_rate  *= self.decay_factor
                epoch_cost = float(allcosts) / len(train_X1)

                #if (epoch % self.saveFrequency == 0 and epoch != 0):
                #if (phase == 3 and epoch - best_saved_epoch > self.saveFrequency and epoch != 0):
                #    saver.save(sess, self.save_path + "/pretrained_lstm.ckpt", global_step=epoch)

                if bestLoss - epoch_cost < self.stop_loss:
                    # print("Exited on epoch  %d, with loss  %.6f comparing with bestLoss: %.6f and stop_loss: %.6f" % (epoch + 1, epoch_cost, bestLoss, epoch_cost))
                    # print('absoulute error difference %f'%( abs( bestLoss - epoch_cost)))
                    pass_best_epochs += 1
                    if pass_best_epochs > self.stopping_iteration:
                        print("Exited on epoch  %d, with loss  %.6f comparing with bestLoss: %.6f and stop_loss: %.6f" % (
                        epoch + 1, epoch_cost, bestLoss, self.stop_loss))
                        if phase < max_phase:
                           phase+=1
                           pass_best_epochs = 0
                           bestLoss = np.inf
                           best_saved_epoch = -10 
                           print ('<<<<<<<<< phase changed from {} to {} >>>>>>>>>>>'.format(phase-1, phase) )
                           continue
                        else:
                           break
                if epoch_cost < bestLoss:
                    if (phase == max_phase  and epoch - best_saved_epoch > self.saveFrequency and epoch != 0):
                        print ('phase: {}, best_loss:{}, epoch:{}, epoch_cost:{}'.format(phase, bestLoss, epoch, epoch_cost) )
                        best_ckpt_saver.handle(epoch_cost, sess, global_step_tensor=tf.constant(epoch))
                        saver.save(sess, self.save_path + "/pretrained_lstm.ckpt", global_step=epoch)
                        best_saved_epoch = epoch
                    pass_best_epochs = 0
                    bestLoss = epoch_cost

                    # if epoch_cost < self.minimum_cost :
                #       print("Exited on epoch  %d, with loss  %.6f" % (epoch + 1, epoch_cost))
                #        break
                print("Epoch = %d, train cost = %.6f"
                      % (epoch + 1, epoch_cost))

            sess.close()

    def predict(self, test_X, bestModel=True, model_path=None):
        if model_path == None:
            model_path = self.save_path
        print('model path is %s' % model_path)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        if not bestModel:
            # checkpoint =  tf.train.latest_checkpoint('data/models/'+entity.split()[0]+'/')
            checkpoint = tf.train.latest_checkpoint(model_path)
        else:
            checkpoint = checkmate.get_best_checkpoint(model_path + "/bestModel/", select_maximum_value=True)
        saver.restore(sess, checkpoint)
        predicted_y = []
        test_X1 = test_X[1]
        test_X2 = test_X[0]
        for i in range(len(test_X1)):
            x1_input = np.array(test_X1[i: i + 1], dtype=np.float64)
            x2_input = np.array(test_X2[i: i + 1], dtype=np.float64)

            yhat = sess.run([self.yhat3], feed_dict={self.X1: x1_input,self.X2: x2_input, self.keep_prob: 1.})
            predicted_y.append(yhat[0][0][0])
        sess.close()
        del test_X
        import gc
        gc.collect()
        return (predicted_y)

    def test(self, test_X, test_y, model_path=None, bestModel=True, write=False):
        if model_path == None:
            model_path = self.save_path
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        # print('data/models/'+entity.split()[0]+'/')
        if not bestModel:
            # checkpoint =  tf.train.latest_checkpoint('data/models/'+entity.split()[0]+'/')
            checkpoint = tf.train.latest_checkpoint(model_path)
        else:
            checkpoint = checkmate.get_best_checkpoint(model_path + "/bestModel/", select_maximum_value=True)
        print('%s checkpoint loaded' % checkpoint)
        saver.restore(sess, checkpoint)

        if write:
            outpFile = open('test_out.csv', 'wb')
            outp = csv.writer(outpFile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            outp.writerow(['county', 'prediction'])

        allcosts = 0
        for i in range(len(test_X)):

            x_input1 = list(map(float, test_X[i: i + 1].values.tolist()[0]))
            x_input1 = np.array([x_input1], dtype=np.float64)

            x_input2 = list(map(float, test_X[i: i + 1].values.tolist()[0]))
            x_input2 = np.array([x_input2], dtype=np.float64)


            y_input = list(map(float, test_y[i: i + 1].values.tolist()[0]))
            y_input = np.array([y_input], dtype=np.float64)

            loss1,loss2,loss3, yhat = sess.run([self.loss1,self.loss2,self.loss3, self.yhat3],
                                  feed_dict={self.X1: x_input1,self.X2: x_input2, self.y: y_input, self.keep_prob: 1.})
            allcosts += loss3
            if write:
                outp.writerow(['1', yhat])
        epoch_cost = float(allcosts) / len(test_X)
        print(epoch_cost)

        if write:
            outpFile.close()
        sess.close()


if __name__ == '__main__':
    table = readfile(input_path)
    mylist = table.split('),')

    write_csv(mylist, output_path)
    pd.set_option('mode.chained_assignment', 'raise')

    data = []
    for row in mylist:
        data.append(row.replace('(', '').split(','))

    df = pd.DataFrame.from_records(data, columns=header)
    out_column_hea = ['01hea_aar']
    out_column_ls = ['ls09_10_avg']
    header.remove('01hea_aar')
    header.remove('ls09_10_avg')
    header.remove('cnty')

    # df.replace({b'NULL': np.nan}, inplace=True)
    df = df.convert_objects(convert_numeric=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10, shuffle=True, random_state=2)
    result = next(kf.split(df), None)
    ind = 0
    while result:
        print('fold %d' % ind)

        train_data = df.iloc[result[0]]
        test_data = df.iloc[result[1]]

        input_data_train = train_data[header][:len(train_data[header]) - 1]
        output_hea_train = train_data[out_column_hea][:len(train_data[header]) - 1]

        input_data_test = test_data[header][:len(test_data[header]) - 1]
        output_hea_test = test_data[out_column_hea][:len(test_data[header]) - 1]

        model = ffNN()
        model.initialize()

        model.train(input_data_train, output_hea_train)
        model.test(input_data_test, output_hea_test)
        result = next(kf.split(df), None)
        ind += 1








