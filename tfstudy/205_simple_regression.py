import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 300).reshape(-1, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

def add_layer(inputs, in_size, out_size, actfun = None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    WX_plus_b = tf.matmul(inputs, W) + b
    if actfun is None:
        outputs= WX_plus_b
    else:
        outputs = actfun(WX_plus_b)

    return outputs

xs = tf.placeholder(shape=[None, 1], dtype=tf.float32)
ys = tf.placeholder(shape=[None, 1], dtype=tf.float32)

l1 = add_layer(xs, 1, 10, actfun=tf.nn.relu)
prediction = add_layer(l1, 10, 1, actfun=None)

loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(ys - prediction),
            reduction_indices = [1]
        )
)

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(3000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        y_pred = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, y_pred, 'r-', lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()
