import tensorflow as tf 
import numpy as np
class siamese:

    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 400])
        self.x2 = tf.placeholder(tf.float32, [None, 400])

        with tf.variable_scope("siamese") as scope:
            self.a1,self.b1,self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.a1,self.b2,self.o2 = self.network(self.x2)
            

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_cds()

    def network(self, x):
        weights = []
        kernel_size =4
        stride = 1
        depth=40
        conv1 = self.conv_layer(x, kernel_size,stride,depth,'conv1')
        conv1r = tf.nn.relu(conv1)
        n_prev_weight = int(x.get_shape()[1])
#        pool1 = tf.layers.max_pooling1d(conv1,4,2,'same')
        conv1_d = tf.reshape(conv1r,[-1, n_prev_weight/stride*depth])
        
        kernel_size =10
        stride = 2
        depth=10
        conv2 = self.conv_layer(conv1_d, kernel_size,stride,depth,'conv2')
        conv2r = tf.nn.relu(conv2)
        n_prev_weight = int(conv1_d.get_shape()[1])
#        pool2 = tf.layers.max_pooling1d(conv2,4,4,'same')
        conv2_d = tf.reshape(conv2r,[-1, n_prev_weight/stride*depth])
        
        fc1 = self.fc_layer(conv1_d, 1500, "fc1")
        ac1 = tf.nn.relu(fc1)
#        fc1_drop = tf.nn.dropout(ac1, 0.1)
        fc2 = self.fc_layer(ac1, 600, "fc2")   
        ac2 = tf.nn.relu(fc2)
#        fc2_drop = tf.nn.dropout(ac2, 0.1)
        fc3 = self.fc_layer(ac2, 200, "fc3")
        return fc1,fc2,fc3
    
    def xavier_init(self,n_inputs, n_outputs, uniform=True):
      if uniform:
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
      else:
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

    def fc_layer(self, bottom, n_weight, name):
        print( bottom.get_shape())
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
#        initer = tf.truncated_normal_initializer(stddev=0.01)
        initer = self.xavier_init(int(n_prev_weight),n_weight)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
#        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.random_uniform([n_weight],-0.001,0.001, dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def conv_layer(self, bottom, kernel_size, stride, depth, name):        
        n_prev_weight = int(bottom.get_shape()[1])
        num_channels = 1 # for 1 dimension
        inputlayer = tf.reshape(bottom, [-1,n_prev_weight,1])
        initer = tf.truncated_normal_initializer(stddev=0.1)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[kernel_size, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.001, shape=[depth*num_channels], dtype=tf.float32))
        
        conv = tf.nn.bias_add( tf.nn.conv1d(inputlayer, W, stride, padding='SAME'), b)
        return conv

    def loss_with_cds(self):
        labels_t = self.y_
        cds = tf.reduce_sum(tf.multiply(self.o1,self.o2),1)
        eucd2 = tf.reduce_mean(tf.pow(tf.subtract(labels_t,cds),2))
        eucd = tf.sqrt(eucd2, name="eucd")
        return eucd



