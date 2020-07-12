import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))
                     
plt.plot(x, y, 'r.')
plt.show()

X = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.zeros([1]))

h = X*W+b

Y = tf.placeholder(tf.float32, shape = [None, 1])
Loss = tf.reduce_mean(tf.square(h - Y))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(Loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training loop
for epoch in range(500):
    _, t_loss = sess.run([optimizer, Loss], feed_dict={X:x, Y:y})
    
    print("Epoch : ", epoch, " Loss : ", t_loss)
    
    if epoch ==0 :
        y_pred = sess.run(h, feed_dict={X:x})
        plt.plot(x, y, 'r.')
        plt.plot(x, y_pred, 'b.')
        plt.show()
    elif (epoch+1) % 100 == 0 :
        y_pred = sess.run(h, feed_dict={X:x})
        plt.plot(x, y, 'r.')
        plt.plot(x, y_pred, 'b.')
        plt.show()