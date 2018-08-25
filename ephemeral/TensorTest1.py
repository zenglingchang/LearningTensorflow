import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class CNN:
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 1])
        self.ys = tf.placeholder(tf.float32, [None, 1])
        self.l1 = self.add_layer(self.xs, 1, 10, activation_function=tf.nn.relu)
        self.prediction = self.add_layer(self.l1, 10, 1,activation_function=None)
        self.loss =  tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction),reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        self.init = tf.global_variables_initializer()  
        self.sess = tf.Session()
        self.sess.run(self.init)
        
    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else :
            outputs = activation_function(Wx_plus_b)
        return outputs
    
    def loop(self, times, x_data, y_data):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x_data, y_data)
        plt.ion()#本次运行请注释，全局运行不要注释
        plt.show()
        for i in range(times):
            self.sess.run(self.train_step, feed_dict={self.xs: x_data, self.ys: y_data})
            if i % 50 == 0:
                try:
                    ax.lines.remove(lines[0])   
                except Exception:
                    pass
                prediction_value = self.sess.run(self.prediction, feed_dict={self.xs: x_data})
                # plot the prediction
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.1)
if __name__ == '__main__':
    x_data = np.linspace(-1,1,300,dtype = np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise
    cnn = CNN()
    cnn.loop(2000, x_data, y_data)