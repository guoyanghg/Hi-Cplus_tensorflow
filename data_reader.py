
import sys
import numpy as np
import os
import gzip
import tensorflow as tf
import math



conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

down_sample_ratio = 16
learning_rate = 0.00001
epochs = 10
HiC_max_value = 100

low_resolution_samples = np.load(gzip.GzipFile('./data/GM12878_replicate_down16_chr19_22.npy.gz', "r")) * down_sample_ratio
high_resolution_samples = np.load(gzip.GzipFile('./data/GM12878_replicate_original_chr19_22.npy.gz', "r"))


low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

sample_num = low_resolution_samples.shape[0]
sample_size = low_resolution_samples.shape[-1]
print(low_resolution_samples.shape)
print(math.pow(121,2))
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = int(padding / 2)
print(half_padding)

Y = []
for i in range(high_resolution_samples.shape[0]):
    no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
    Y.append(no_padding_sample)
Y = np.array(Y).astype(np.float32)
Y_data = Y.reshape((Y.shape[0], -1))
X_data = low_resolution_samples.transpose(0,2,3,1)



def model(X):
    conv2d1 = tf.layers.conv2d(
        inputs=X,
        filters=conv2d1_filters_numbers,
        kernel_size=[conv2d1_filters_size, conv2d1_filters_size],
        padding='valid',
        activation=tf.nn.relu,
    )
    conv2d2 = tf.layers.conv2d(
        inputs=conv2d1,
        filters=conv2d2_filters_numbers,
        kernel_size=[conv2d2_filters_size, conv2d2_filters_size],
        padding='valid',
        activation=tf.nn.relu,
    )
    conv2d3 = tf.layers.conv2d(
        inputs=conv2d2,
        filters=conv2d3_filters_numbers,
        kernel_size=[conv2d3_filters_size, conv2d3_filters_size],
        padding='valid',
        activation=tf.nn.relu,
    )

    return conv2d3


X = tf.placeholder("float", [None, sample_size, sample_size, 1], name='X')
Y = tf.placeholder("float", [None, conv2d3_filters_numbers*28*28])

conv3 = model(X)
flat_conv3 = tf.reshape(conv3, [-1, conv2d3_filters_numbers * 28 * 28])
loss = tf.reduce_mean(tf.square(flat_conv3-Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver=tf.train.Saver()
tf.add_to_collection('Y_pred',conv3)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 使用start_queue_runners之后，才会开始填充队列
    for j in range(epochs):
        for i in range(sample_num):
            [t,l]= sess.run([train_op,loss],feed_dict={X:[X_data[i]],Y:[Y_data[i]]})
            print(l)
        print("epochs%d is done!",j)

    print('training done')
    saver.save(sess,'model/test_model')
    print('model saved..')



