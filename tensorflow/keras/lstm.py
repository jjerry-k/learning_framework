import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# Loading Mnist Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
tf.set_random_seed(777)

# Setting hyper parameter
learning_rate = 0.001
total_epoch = 30
batch_size = 128
n_input = 28
n_step = 28
n_hidden1 = 128
n_class = 10


X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden1, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# LSTM cell 선언.
# RNN을 쓰고 싶으면 BasicRNNCell로 바꾸면 됨.
# Stacked LSTM을 하고 싶으면 cell2 선언.
cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden1)
#cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden1)

# 선언된 LSTM cell, X를 이용하여 네트워크 생성.
outputs_1, states_1 = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32,scope="LSTM1")

# Stacked LSTM을 하고 싶으면 다음과 같이 선언.
#outputs_2, states_2 = tf.nn.dynamic_rnn(cell2, outputs_1, dtype=tf.float32, scope="LSTM2")

# LSTM -> Fully Connected Layer -> Classification

# outputs_1 : [ ? , num_step, num_hidden
#             -> [num_step, ? , num_hidden]
outputs = tf.transpose(outputs_1, [1, 0, 2])

# Sequence의 마지막 출력값
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# GPU 메모리 할당.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)


        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch : %04d'% (epoch + 1),'Avg cost : {:f}'.format(total_cost / total_batch))

print('Optimization Done')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('Test Accuracy : ', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))