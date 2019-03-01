import tensorflow as tf
import pandas as pd
import csv
import numpy as np
import pickle 
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


def init_weights(shape,trainable=True, stddev=0.5,name=None,mean=0.):
    """ Weight initialization """
    if name is None:
       weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev, mean=mean),trainable=trainable)
       b = tf.Variable(tf.zeros(1, dtype=tf.float32),trainable=trainable)
    else:
       weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev, mean=mean),trainable=trainable,name='w_'+name) 
       b = tf.Variable(tf.zeros(1, dtype=tf.float32),trainable=trainable,name='b_'+name)
    return weights, b


class ffNN():
    tf.reset_default_graph()

    def __init__(self, hidden_nodes=[[8,2],[32,8]], epochs=3, learning_rate=0.1, saveFrequency=1,
                 save_path='./models/ControlOnly', decay=False, decay_step=10, decay_factor=0.7,
                 stop_loss=0.0001, regularization_factor=[0.05,0.04,0.03,0.05], keep_probability=0.7, minimum_cost=0.2,
                 activation_function='sigmoid', batch_size=1,shuffle=True,optimizer='Adam',stopping_iteration=[10,10,1,1], stddev=[0.5,0.1],max_phase=4,start_phase=1,FA=False,RC=False,combine_model='yhat', use_dev=False):
        self.hidden_nodes = hidden_nodes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.saveFrequency = saveFrequency
        self.save_path = save_path
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
        self.stopping_iteration = stopping_iteration
        self.stddev=stddev
        self.max_phase=max_phase
        self.start_phase=start_phase
        self.FA=FA
        self.RC=RC
        self.Dev = use_dev
        self.combine_model=combine_model
        print('model started working :D')

    def forwardprop(self, X, w_in, b_in, keep_prob,activation_func):
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
        #z_norm = tf.layers.batch_normalization(z_norm, training=is_train)
        if activation_func == 'relu':
            h = tf.nn.relu(z_norm)
        elif activation_func == 'sigmoid':
            h = tf.nn.sigmoid(z_norm)
        elif activation_func == 'tanh':
            h = tf.nn.tanh(z_norm)
        else:
            h = z_norm
        # h = tf.nn.sigmoid(tf.matmul(X, w_in))  # The \sigma function
        # h = tf.matmul(X, w_1)
        # h = tf.add(h, b_in)
        # if self.keep_prob <> 1:
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        return h
    def forward_pass(self, input_x, phase, reg_factor, keep_prob, x_size, y_size, learning_rate, w_in=None ,b_in= None,prefix='',mean=0., rg_method='L2') :
            if w_in is None:
                w_in, b_in = init_weights((x_size, self.hidden_nodes[phase-1][0]), stddev=self.stddev[phase-1],name='in0_phase'+str(phase)+prefix,mean=mean)
            h_out = self.forwardprop(input_x, w_in, b_in, keep_prob,self.activation_function[0])
            #l1_norm = tf.add_n([tf.reduce_sum(tf.abs(w_in)), tf.reduce_sum(tf.abs(b_in))])
            l_norm =  tf.add_n([tf.nn.l2_loss(w_in), tf.nn.l2_loss(b_in)]) if rg_method=='L2' else tf.add_n([tf.reduce_sum(tf.abs(w_in)), tf.reduce_sum(tf.abs(b_in))])  
            # Forward propagation
            for i in range(len(self.hidden_nodes[phase-1]) - 1):
                w_in, b_in = init_weights((self.hidden_nodes[phase-1][i], self.hidden_nodes[phase-1][i + 1]), stddev=self.stddev[phase-1],name='in'+str(i+1)+'_phase'+str(phase)+prefix, mean=mean)
                h_out = self.forwardprop(h_out, w_in, b_in, keep_prob,self.activation_function[1])
                l_norm = tf.add_n([tf.nn.l2_loss(w_in), tf.nn.l2_loss(b_in),l_norm]) if rg_method=='L2' else l_norm
                #l2_norm1 = tf.add(tf.nn.l2_loss(b_in1), l2_norm1)
            w_out, b_out = init_weights((self.hidden_nodes[phase-1][-1], y_size), stddev=self.stddev[phase-1],name='out_phase'+str(phase)+prefix,mean=mean)
            l_norm = tf.add_n([tf.nn.l2_loss(w_out),tf.nn.l2_loss(b_out), l_norm]) if rg_method=='L2' else l_norm
            yhat = tf.add(tf.matmul(h_out, w_out), b_out)
            # Backward propagation
            mse = tf.losses.mean_squared_error(labels=self.y, predictions=yhat)
            #l2_norm1 = tf.Print(l2_norm1,[l2_norm1],message='l2_norm1')

            #loss = tf.add(mse,reg_factor * l1_norm)
            loss = tf.add(mse,reg_factor * l_norm)

            if self.decay == 1 or self.decay is True:
                 learning_rate_adam = tf.train.exponential_decay(learning_rate,self.global_step, self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
                 learning_rate_adam = learning_rate
            updates = tf.train.AdamOptimizer(learning_rate=learning_rate_adam).minimize(loss,var_list=[v for v in tf.trainable_variables() if ('phase'+str(phase)+prefix) in v.name])
            return yhat, mse, loss, updates, h_out, learning_rate_adam,l_norm, w_out, b_out



    def initialize(self, x1_size=11,x2_size=2000,xA_size=22000,x2n_size=20000, y_size=1):

        print('initializing parameters')
        self.X1 = tf.placeholder("float", shape=[None, x1_size], name='X1Value')
        self.X2 = tf.placeholder("float", shape=[None, x2_size], name='X2Value')
        self.XA = tf.placeholder("float", shape=[None, xA_size], name='XAValue')
        self.X2n = tf.placeholder("float", shape=[None, x2n_size], name='XnValue')

        self.learning_rate1 = tf.placeholder("float", shape=(), name='learning_rate1')
        self.learning_rate2 = tf.placeholder("float", shape=(), name='learning_rate2')
        self.learning_rate3 = tf.placeholder("float", shape=(), name='learning_rate3')
        self.learning_rate4 = tf.placeholder("float", shape=(), name='learning_rate4')
        self.learning_rate5 = tf.placeholder("float", shape=(), name='learning_rate5')

        self.reg_factor1 =tf.placeholder("float", name='reg_factor1',shape=())
        self.reg_factor2 =tf.placeholder("float", name='reg_factor2',shape=())
        self.reg_factor3 =tf.placeholder("float", name='reg_factor3',shape=())
        self.reg_factor4 =tf.placeholder("float", name='reg_factor4',shape=())
        self.reg_factor5 =tf.placeholder("float", name='reg_factor_ngram',shape=())

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
        with tf.variable_scope("initialization", reuse=tf.AUTO_REUSE):
            # self.reg_factor = tf.get_variable( name='reg_factor',shape=(), dtype = tf.float32)
            #self.reg_factor = self.regularization_factor
            self.increment_global_step = tf.assign_add(self.global_step,1,name = 'increment_global_step')

            ############################# PHASE 1 ########################

            self.yhat1, self.mse1, self.loss1, self.updates1, self.h_out1, self.learning_rate1_adam,self.l2_norm1, _ ,_ =self.forward_pass(input_x=self.X1, phase=1,reg_factor=self.reg_factor1,keep_prob=self.keep_prob,x_size=x1_size,y_size=y_size, learning_rate=self.learning_rate1 ) 
            ############################# PHASE 2 #######################

            self.yhat2, self.mse2, self.loss2, self.updates2t , self.h_out2, self.learning_rate2_adam,self.l2_norm2, _,_ =self.forward_pass(input_x=self.X2, phase=2,reg_factor=self.reg_factor2,keep_prob=self.keep_prob,x_size=x2_size,y_size=y_size, learning_rate=self.learning_rate2 )


            #self.yhat2n, self.mse2n, self.loss2n, self.updates2n , h_out2n, _,l2_norm2n,_,_ =self.forward_pass(input_x=self.X2n,phase=2,reg_factor=self.reg_factor5,keep_prob=self.keep_prob,x_size=x2n_size,y_size=y_size, learning_rate=self.learning_rate2, prefix='ngram')


            self.updates2 = self.updates2t
            #self.updates2 = tf.group(self.updates2t,self.updates2n)




            ############### FA  ###############
            if self.FA:
               #self.XAdapted = tf.multiply(tf.reshape(self.X1,[-1,1,x1_size]), tf.reshape(self.X2,[-1,x2_size,1]))
               #self.XAdapted = tf.reshape(self.XAdapted,[-1,x2_size * x1_size]) 
               #XAdapted_size = x2_size * x1_size
               '''
               self.yhatA, self.mseA, self.lossA, self.updatesA , h_outA, self.learning_rate2_adam,self.l2_normA, _, _ =self.forward_pass(input_x=self.XA,phase=2,reg_factor=self.reg_factor2,keep_prob=self.keep_prob,x_size=xA_size,y_size=y_size, learning_rate=self.learning_rate2, prefix='FA',rg_method='L2')
               #self.updates2 = self.updatesA
               self.updates2 = tf.group(self.updates2t,self.updatesA)
               #self.yhat2 = self.yhatA
               self.loss2 =  self.lossA + self.loss2
               self.mse2 = self.mseA + self.mse2
               self.l2_norm2 =  self.l2_normA +self.l2_norm2 
               '''
               self.yhat3, self.mse3, self.loss3, self.updates3 , h_out3, self.learning_rate3_adam,self.l2_norm3, _, _ =self.forward_pass(input_x=self.XA,phase=3,reg_factor=self.reg_factor2,keep_prob=self.keep_prob,x_size=xA_size,y_size=y_size, learning_rate=self.learning_rate2,rg_method='L2')
               '''
               self.XAdapted = tf.multiply(tf.reshape(self.X1,[-1,1,x1_size]), tf.reshape(self.X2,[-1,x2_size,1]))
               self.XAdapted = tf.reshape(self.XAdapted,[-1,x2_size * x1_size]) 
               XAdapted_size = x2_size * x1_size
               w_inA, b_inA = init_weights((XAdapted_size, self.hidden_nodes[1][0]), stddev=self.stddev[1],name='inA10')
               h_outA = self.forwardprop(self.XAdapted, w_inA, b_inA, self.keep_prob,self.activation_function[0])
               l2_normA = tf.add(tf.nn.l2_loss(w_inA), l2_normA)
               l2_normA = tf.add(tf.nn.l2_loss(b_inA), l2_normA)
   
               ## Forward propagation
               for i in range(len(self.hidden_nodes[1]) - 1):
                   w_inA, b_inA = init_weights((self.hidden_nodes[1][i], self.hidden_nodes[1][i + 1]), stddev=self.stddev[1],name='in1'+str(i+1))
                   h_outA = self.forwardprop(h_outA, w_inA, b_inA, self.keep_prob,self.activation_function[1])
                   l2_normA = tf.add(tf.nn.l2_loss(w_inA), l2_normA)
                   l2_normA = tf.add(tf.nn.l2_loss(b_inA), l2_normA)
               w_outA, b_outA = init_weights((self.hidden_nodes[1][-1], y_size), stddev=self.stddev[1],name='out1')
               l2_normA = tf.add(tf.nn.l2_loss(w_outA), l2_normA)
               l2_normA = tf.add(tf.nn.l2_loss(b_outA), l2_normA)
   
               self.yhatA = tf.add(tf.matmul(h_outA, w_outA), b_outA)
               self.mseA = tf.losses.mean_squared_error(labels=self.y, predictions=self.yhatA)
               self.lossA = tf.add(self.mseA, self.reg_factor22 * l2_normA)
   
               if self.decay == 1 or  self.decay is True:
                   self.learning_rate2_adam = tf.train.exponential_decay(self.learning_rate2,self.global_step,  self.decay_step, self.decay_factor,
                                                          staircase=True)
               else:
                   self.learning_rate2_adam= self.learning_rate2
               self.updatesA = tf.train.AdamOptimizer(learning_rate=self.learning_rate2_adam).minimize(self.lossA)
               self.loss2 = self.lossA
               self.mse2 = self.mseA
               self.yhat2 = self.yhatA
               '''

            ########### END OF FA ###########



            ########################## Phase3 ############################
            #w_in3 = tf.Variable((np.ones([2,1]) * np.array([[0.5], [0.5]])).astype(np.float32 ),name='w_in_phase3')
            #b_in3 = tf.Variable(tf.zeros(1, dtype=tf.float32),name='b_in_phase3')

            if self.combine_model == 'yhat':
                outputs = [self.yhat1, self.yhat2]#, self.yhat2A] 
                if self.FA :
                    outputs+= [self.yhat3]
                combinationW = [[0.9], [0.1], [0.1]]
                combinationW = combinationW[:len(outputs)]
                self.w_out4 = tf.Variable((np.ones([len(outputs),1]) * np.array(combinationW)).astype(np.float32 ),name='w_out_phase4')
                self.b_out4 = tf.Variable(tf.zeros(1, dtype=tf.float32),name='b_out_phase4')
                self.yhat4 = self.forwardprop(tf.concat(outputs,axis=1), self.w_out4, self.b_out4, self.keep_prob,self.activation_function[2]) 
                #self.yhat3, self.mse3, self.loss3, self.updates3 , h_out3, self.learning_rate3_adam,l2_norm3,self.w_out3,self.b_out3 =self.forward_pass(input_x=tf.concat([self.yhat1, self.yhat2],axis=1),phase=3,reg_factor=self.reg_factor3,keep_prob=self.keep_prob,x_size=last_layer_nodes,y_size=y_size, learning_rate=self.learning_rate3,mean=0.5)#,mean1=0.05,mean2=0.5)#, w_in=w_in3, b_in=b_in3 )
                #self.yhat3, self.mse3, self.loss3, self.updates3 , h_out3, self.learning_rate3_adam,l2_norm3,self.w_out3,self.b_out3 =self.forward_pass(input_x=tf.concat([self.yhat1, self.yhat2,self.yhat2n],axis=1),phase=3,reg_factor=self.reg_factor3,keep_prob=self.keep_prob,x_size=last_layer_nodes,y_size=y_size, learning_rate=self.learning_rate3)#,mean1=0.05,mean2=0.5)#, w_in=w_in3, b_in=b_in3 )
            else:
                last_layer_nodes = self.hidden_nodes[0][-1]+self.hidden_nodes[1][-1]#+self.hidden_nodes[1][-1] #2
                ##self.yhat3, self.mse3, self.loss3, self.updates3 , h_out3, self.learning_rate3_adam,l2_norm3 =self.forward_pass(input_x=tf.concat([self.h_out1, self.h_out2,h_out2n],axis=1),phase=3,reg_factor=self.reg_factor3,keep_prob=self.keep_prob,x_size=last_layer_nodes,y_size=y_size,learning_rate=self.learning_rate3)
                self.yhat3, self.mse3, self.loss3, self.updates3 , h_out3, self.learning_rate3_adam,l2_norm3, self.w_out3, self.b_out3=self.forward_pass(input_x=tf.concat([self.h_out1, self.h_out2],axis=1),phase=3,reg_factor=self.reg_factor3,keep_prob=self.keep_prob,x_size=last_layer_nodes,y_size=y_size, learning_rate=self.learning_rate3, mean=0.01 )

            # Backward propagation
            self.mse4 = tf.losses.mean_squared_error(labels=self.y, predictions=self.yhat4)
            self.loss4 = self.mse4 #tf.add(self.mse3,self.reg_factor* l2_norm3)
            
            if self.decay == 1 or  self.decay is True:
               self.learning_rate4_adam = tf.train.exponential_decay(self.learning_rate4,self.global_step,  self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
               self.learning_rate4_adam = self.learning_rate4
            
            self.updates4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate4_adam).minimize(self.loss4,var_list=[v for v in tf.trainable_variables() if 'phase4' in v.name])





            ############################## Phase4 #####################################
            if self.RC:
                 self.loss4 = tf.add_n([0.7*self.mse3,0.2*self.reg_factor4* self.l2_norm2,0.3*self.mse2])
                 self.mse4 = tf.add(0.7*self.mse3,0.3*self.mse2)

            else:
                 self.loss5 = tf.add_n([self.mse4,self.reg_factor4* self.l2_norm2, self.reg_factor4* self.l2_norm3])
                 self.mse5 = self.mse4


            #self.loss4 = tf.add(self.mse3,self.reg_factor4* tf.add_n([self.l2_norm2,self.l2_norm1]))
            #self.mse4 = tf.Print(self.mse4,[self.mse3,self.loss4,self.reg_factor4, l2_norm2,l2_norm1],message="mse, loss,self.reg_factor4, l2_norm2,l2_norm1",summarize=20)
            self.yhat5 = self.yhat4
            if self.decay == 1 or  self.decay is True:
               self.learning_rate5_adam = tf.train.exponential_decay(self.learning_rate5,self.global_step,  self.decay_step, self.decay_factor,
                                                       staircase=True)
            else:
               self.learning_rate5_adam = self.learning_rate5
            

            self.updates5 = tf.train.AdamOptimizer(learning_rate=self.learning_rate5_adam).minimize(self.loss5)#,var_list=[v for v in tf.trainable_variables() if 'phase2' or 'phase3' in v.name])
            #self.updates4L = tf.train.AdamOptimizer(learning_rate=self.learning_rate4_adam).minimize(self.loss4,var_list=[v for v in tf.trainable_variables() if 'phase2' or 'phase3' in v.name])
            #self.updates4C = tf.train.AdamOptimizer(learning_rate=self.learning_rate4_adam).minimize(self.loss4,var_list=[v for v in tf.trainable_variables() if 'phase1' in v.name])
            #self.updates4 = tf.group(self.updates4C, self.updates4L)
            



            #self.final_opt = tf.group(self.updates1,self.updates2,self.updates3,self.updates4)
           
    def apply_decay(self, lr, epoch ):
        lr = tf.Print(lr , [lr, lr * self.decay_factor, epoch , self.decay_step, epoch % self.decay_step , self.decay ],message='lr inside apply_decay: ')
        if self.decay==1 and epoch % self.decay_step ==0:
           #lr = tf.Print(lr , [lr, lr * self.decay_factor, epoch , self.decay_step, epoch % self.decay_step , self.decay ],message='lr inside apply_decay: ')
           return lr * self.decay_factor           
        else:
           return lr

    def splitShuffle(self, input_set,devSize, shuffle=True):
            s = np.arange(0, len(input_set[0]), 1)
            if shuffle:
                np.random.shuffle(s)
            devOrdering = s[0: int(len(s)*devSize)]
            trainOrdering = s[int(len(s)*devSize):]
            train_set = [ arr[trainOrdering] for arr in input_set]
            dev_set = [arr[devOrdering] for arr in input_set]
            return train_set, dev_set

    def train(self, X, y, Xtest, ytest):
            #tf.reset_default_graph()
            print('x_1 size: ', X[0].shape, ' x_2 size: ',  X[1].shape, ', x_3 size: ', X[2].shape)
            self.desired_epoch_cost = np.inf
            # print('y_2 size %s %s' % y.shape)
            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(save_relative_paths=True,max_to_keep = 300)
            bestLoss = np.inf
            import os
            import shutil
            best_save_dir = self.save_path + '/bestModel/'
            if os.path.exists(best_save_dir):
                shutil.rmtree(best_save_dir)
            os.makedirs(best_save_dir)
            best_ckpt_saver = BestCheckpointSaver(
                save_dir=best_save_dir,
                num_to_keep=1,
                maximize=False
            )
            X2 = X[0]
            X1 = X[len(X)-1] if not self.FA else X[int(len(X)/2)-1] 
            XAdapted = X[len(X)-2]
            X2n = X[len(X)-2]
            pass_best_epochs = 0
            best_saved_epoch = -10
            phase = self.start_phase #1
            max_phase = self.max_phase # 1
            learning_rates_adam = []


            all_epoch_cost = []
            all_epoch_MSE = []
            if self.Dev:
                [train_x1,train_x2,train_x2n,train_adaptedx,train_y],[train_x1_dev,train_x2_dev,train_x2n_dev,train_adaptedx_dev,train_y_dev]= self.splitShuffle([X1,X2,X2n,XAdapted,y],devSize=0.1, shuffle=self.shuffle)
                train_y_dev = np.transpose(np.array([train_y_dev],dtype=np.float64))



            for epoch in range(self.epochs):
                MSE = 0.
                allcosts = 0
                weights = []
                l2_norm = 0.
                weight_b = -100
                if phase == 2 or phase == 1:
                   keep_prob = self.keep_probability[0]
                else:
                   keep_prob = self.keep_probability[1]


                if self.Dev:
                   [train_x1,train_x2,train_x2n,train_adaptedx,train_y], _ = self.splitShuffle([train_x1,train_x2,train_x2n,train_adaptedx,train_y],devSize=0, shuffle=self.shuffle)
                else:
                   [train_x1,train_x2,train_x2n,train_adaptedx,train_y], _  =self.splitShuffle([X1,X2,X2n,XAdapted,y],devSize=0., shuffle=self.shuffle)

                #[train_x1,train_x2,train_x2n,train_adaptedx,train_y],[train_x1_dev,train_x2_dev,train_x2n_dev,train_adaptedx_dev,train_y_dev]= self.splitShuffle([X1,X2,X2n,XAdapted,y],devSize=0.33, shuffle=self.shuffle)
                #train_y_dev = np.transpose(np.array([train_y_dev],dtype=np.float64))

                # print(train_x)
                sess.run(self.increment_global_step) 
                for i in range(0, len(train_x1) - self.batch_size + 1, self.batch_size):
                    # x_input = list(map(float,train_x[i: i + 2].values.tolist()[0]))
                    # x_input = np.array([x_input],dtype=np.float64)
                    x1_input = np.array(train_x1[i: i + self.batch_size], dtype=np.float64)
                    x2_input = np.array(train_x2[i: i + self.batch_size], dtype=np.float64)
                    x2n_input = np.array(train_x2n[i: i + self.batch_size], dtype=np.float64)
                    xa_input = np.array(train_adaptedx[i: i + self.batch_size], dtype=np.float64)
                    y_input = np.transpose(np.array([train_y[i: i + self.batch_size]],dtype=np.float64)) 

                    feed_dict={self.reg_factor1: self.regularization_factor[0],self.reg_factor2: self.regularization_factor[1],self.reg_factor3: self.regularization_factor[2],self.reg_factor4: self.regularization_factor[3],self.reg_factor5: self.regularization_factor[4],self.X1: x1_input, self.X2: x2_input,self.X2n: x2n_input, self.XA: xa_input, self.y: y_input, self.keep_prob: keep_prob,self.learning_rate1:self.learning_rate[0] ,self.learning_rate2:self.learning_rate[1]  , self.learning_rate3:self.learning_rate[2] ,self.learning_rate4:self.learning_rate[3], self.learning_rate5:self.learning_rate[4]}#,self.random_mask: random_mask   }

                    if  phase >1 and (self.FA or self.FA ==1) :
                          loss = sess.run([eval('self.updates'+str(phase)),eval('self.loss'+str(phase)),eval('self.learning_rate'+str(phase)+'_adam'),eval('self.mse'+str(phase)),eval('self.yhat'+str(phase)), self.w_out4],feed_dict= feed_dict)
                          #w_in_shape= tf.shape(loss[7])
                    else: 
                          loss = sess.run([eval('self.updates'+str(phase)),eval('self.loss'+str(phase)),eval('self.learning_rate'+str(phase)+'_adam'),eval('self.mse'+str(phase)),eval('self.yhat'+str(phase)), self.w_out4],feed_dict=feed_dict)
                    '''
                    if  phase == 4:
                          loss = sess.run([eval('self.updates'+str(phase)),eval('self.loss'+str(phase)),eval('self.learning_rate'+str(phase)+'_adam'),eval('self.mse'+str(phase)),eval('self.yhat'+str(phase)),self.w_out3,self.b_out3,self.l2_norm2],feed_dict={self.reg_factor1: self.regularization_factor[0],self.reg_factor2: self.regularization_factor[1],self.reg_factor3: self.regularization_factor[2],self.reg_factor4: self.regularization_factor[3],self.reg_factor5: self.regularization_factor[4],self.X1: x1_input, self.X2: x2_input,self.X2n: x2n_input, self.y: y_input, self.keep_prob: keep_prob,self.learning_rate1:self.learning_rate[0] ,self.learning_rate2:self.learning_rate[1]  , self.learning_rate3:self.learning_rate[2] ,self.learning_rate4:self.learning_rate[2]   })                    
                          weights = loss[5]
                          weight_b = loss[6]
                          l2_norm = loss[7]
                    else: 
                          loss = sess.run([eval('self.updates'+str(phase)),eval('self.loss'+str(phase)),eval('self.learning_rate'+str(phase)+'_adam'),eval('self.mse'+str(phase)),eval('self.yhat'+str(phase))],feed_dict={self.reg_factor1: self.regularization_factor[0],self.reg_factor2: self.regularization_factor[1],self.reg_factor3: self.regularization_factor[2],self.reg_factor4: self.regularization_factor[3],self.reg_factor5: self.regularization_factor[4],self.X1: x1_input, self.X2: x2_input,self.X2n: x2n_input, self.y: y_input, self.keep_prob: keep_prob,self.learning_rate1:self.learning_rate[0] ,self.learning_rate2:self.learning_rate[1]  , self.learning_rate3:self.learning_rate[2] ,self.learning_rate4:self.learning_rate[2]   })                    
                    '''
                    
                    #self.last_learning_rate = loss[7]
                    allcosts += loss[1]
                    learning_rates_adam = loss[2]
                    MSE +=  loss[3]
                    
                    yhats = loss[4]
                     
                    weights = loss[5]
                    #weight_b = loss[6]
                # if self.decay is True and epoch % self.decay_step ==0:
                #    self.learning_rate  *= self.decay_factor
                epoch_cost = float(allcosts) * self.batch_size / len(train_x1)
                epoch_MSE = float(MSE) * self.batch_size / len(train_x1)
                #if (epoch % self.saveFrequency == 0 and epoch != 0):
                #if (phase == 3 and epoch - best_saved_epoch > self.saveFrequency and epoch != 0):
                #    saver.save(sess, self.save_path + "/pretrained_lstm.ckpt", global_step=epoch)


                X2test = Xtest[0]
                X1test = Xtest[len(Xtest)-1] if not self.FA else Xtest[int(len(Xtest)/2)-1]
                XAdaptedtest = Xtest[len(Xtest)-2]
                X2ntest = Xtest[len(Xtest)-2]
                ytest_feed = np.transpose(np.array([ytest],dtype=np.float64))
                feed_dict_test = {self.reg_factor1: self.regularization_factor[0],self.reg_factor2: self.regularization_factor[1],self.reg_factor3: self.regularization_factor[2],self.reg_factor4: self.regularization_factor[3],self.reg_factor5: self.regularization_factor[4], self.X1: X1test, self.X2: X2test, self.X2n: X2ntest, self.XA:XAdaptedtest, self.y: ytest_feed, self.keep_prob: 1.}
                costTest,MSETest = sess.run([eval('self.loss'+str(phase)),eval('self.mse'+str(phase))], feed_dict=feed_dict_test)

                print("Phase= %d, Epoch = %d, train cost = %.6f, train mse:%.6f,test MSE = %.6f , weights: %s"#, weights: %s , weight_b: %.6f, l2norm3: %.6f"
                      % (phase, epoch + 1, epoch_cost, epoch_MSE,MSETest, ', '.join(map(str, weights))) ) #, weight_b,l2_norm) )
 



                if epoch_cost < bestLoss:
                    if (epoch - best_saved_epoch > self.saveFrequency and epoch != 0):
                       bestLoss = epoch_cost
                       pass_best_epochs = 0
                       #if (epoch - best_saved_epoch > self.saveFrequency and epoch != 0):
                       if (phase == max_phase):
                         print ('phase: {}, best_loss:{}, epoch:{}, epoch_cost:{}'.format(phase, bestLoss, epoch, epoch_cost) )
                         best_ckpt_saver.handle(epoch_cost, sess, global_step_tensor=tf.constant(epoch))
                         saver.save(sess, self.save_path + "/pretrained_lstm.ckpt", global_step=epoch)
                         #saver.save(sess, self.save_path +'_'+str(phase)+"/pretrained_lstm.ckpt",global_step=epoch)
                         best_saved_epoch = epoch

                if self.RC:
                  #### Active for RC Model
                  if phase == 2 or phase == 3:
                    sess.run(self.initialize_global_step)
                    pass_best_epochs = 0 
                    print ('phase: {}, best_loss:{}, epoch:{}, epoch_cost:{}'.format(phase, bestLoss, epoch, epoch_cost) )
                    best_ckpt_saver.handle(epoch_cost, sess, global_step_tensor=tf.constant(epoch))
                    saver.save(sess, self.save_path +'_'+str(phase)+"/pretrained_lstm.ckpt",global_step=epoch)
                    phase +=1
                    bestLoss = np.inf
                    continue #if bestLoss - epoch_cost < self.stop_loss  or ( epoch_cost < 0.75 * self.desired_epoch_cost and self.desired_epoch_cost is not np.inf):
                if bestLoss - epoch_cost < self.stop_loss:
                    pass_best_epochs += 1
                    if  pass_best_epochs > self.stopping_iteration[phase-1]:
                        print("Exited on epoch  %d, with loss  %.6f comparing with bestLoss: %.6f and stop_loss: %.6f" % (
                        epoch + 1, epoch_cost, bestLoss, self.stop_loss))
                        self.desired_epoch_cost = bestLoss if self.desired_epoch_cost is np.inf else min(self.desired_epoch_cost , bestLoss )
                        if phase < max_phase:
                           sess.run(self.initialize_global_step)
                           
                           pass_best_epochs = 0
                           bestLoss = np.inf
                           best_saved_epoch = -10 
                           #if phase == 3:
                           #    learning_rate_phase3 = (self.last_learning_rate + self.learning_rate[2])/2
                           print ('<<<<<<<<< phase changed from {} to {} >>>>>>>>>>>'.format(phase, phase+1) )
                           saver.save(sess, self.save_path +'_'+str(phase)+"/pretrained_lstm.ckpt",global_step=epoch)
                           print('saved to: ' , self.save_path +'_'+str(phase)+"/pretrained_lstm.ckpt")
                           phase +=1        
                           continue
                        else:
                           saver.save(sess, self.save_path +"/pretrained_lstm.ckpt",global_step=epoch)
                           break
                #if epoch_cost < bestLoss:
                    #pass_best_epochs = 0
                    #bestLoss = epoch_cost
                    # if epoch_cost < self.minimum_cost :
                #       print("Exited on epoch  %d, with loss  %.6f" % (epoch + 1, epoch_cost))
                #        break
                #print("Phase= %d, Epoch = %d, train cost = %.6f, train mse:%.6f"#, weights: %s , weight_b: %.6f, l2norm3: %.6f"
                #      % (phase, epoch + 1, epoch_cost, epoch_MSE))#, ', '.join(map(str, weights)), weight_b,l2_norm) )
            num_parameters = sum([np.prod(tvar.get_shape().as_list())
                          for tvar in tf.trainable_variables()])
            print('total number of parameters: %d'%num_parameters)
            sess.close()

    def predict(self, test_X, bestModel=True, model_path=None,phase=None, reset_graph=False):
        if model_path == None:
            model_path = self.save_path
        sess = tf.InteractiveSession()
        saver = tf.train.Saver(save_relative_paths=True,max_to_keep = 300)
        if not bestModel:
            # checkpoint =  tf.train.latest_checkpoint('data/models/'+entity.split()[0]+'/')
            load_path = model_path if phase is None else model_path+'_'+str(phase)
            phase = min(self.max_phase,3) if phase is None else phase 
            checkpoint = tf.train.latest_checkpoint(load_path)
            print('model path: %s' %(checkpoint))
            
        else:
            checkpoint = checkmate.get_best_checkpoint(model_path + "/bestModel/", select_maximum_value=True)
            print('model path: %s' % (checkpoint))
        saver.restore(sess, checkpoint)
        predicted_y = []
        test_X1 = test_X[len(test_X)-1] if not self.FA else test_X[int(len(test_X)/2)-1] # control input
        test_X2 = test_X[0] # topic input
        test_X2n = test_X[len(test_X)-2] # ngram input
        test_XA = test_X[len(test_X)-2] # adaptation input
        '''
        for i in range(len(test_X1)):
            x1_input = np.array(test_X1[i: i + 1], dtype=np.float64)
            x2_input = np.array(test_X2[i: i + 1], dtype=np.float64)
            x2n_input = np.array(test_X2n[i: i + 1], dtype=np.float64)
            xA_input = np.array(test_XA[i: i + 1], dtype=np.float64)

            #yhat = sess.run([self.yhat2], feed_dict={self.X1: x1_input,self.X2: x2_input, self.keep_prob: 1.})
            ###yhat = sess.run(eval('self.yhat'+str(phase)), feed_dict={self.X1: x1_input,self.X2: x2_input,self.X2n: x2n_input, self.keep_prob: 1.})
            yhat = sess.run(eval('self.yhat'+str(phase)), feed_dict={self.X1: x1_input,self.X2: x2_input,self.XA: xA_input,self.X2n: x2n_input, self.keep_prob: 1.})

            #yhat = sess.run(eval('self.yhat'+str(phase+1)), feed_dict={self.X1: x1_input,self.X2: x2_input,self.XA: xA_input, self.keep_prob: 1.})
            predicted_y.append(yhat[0][0])
        '''
        x1_input = np.array(test_X1, dtype=np.float64)
        x2_input = np.array(test_X2, dtype=np.float64)
        x2n_input = np.array(test_X2n, dtype=np.float64)
        xA_input = np.array(test_XA, dtype=np.float64)
        yhat = sess.run(eval('self.yhat'+str(phase)), feed_dict={self.X1: x1_input,self.X2: x2_input,self.XA: xA_input,self.X2n: x2n_input, self.keep_prob
: 1.})
        predicted_y = yhat

        sess.close()
        del test_X
        import gc
        gc.collect()
        if reset_graph:
            tf.reset_default_graph()
        return [predicted_y]

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








