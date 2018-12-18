
import sys
import numpy as np
import os
import gzip
import tensorflow as tf



down_sample_ratio = 16
low_resolution_samples = np.load(gzip.GzipFile('./data/GM12878_replicate_down16_chr19_22.npy.gz', "r")) * down_sample_ratio
low_resolution_samples = low_resolution_samples.transpose(0,2,3,1)
print(low_resolution_samples.shape)


with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('model/test_model.meta')
    new_saver.restore(sess,'model/test_model')
    Y = tf.get_collection('Y_pred')[0]
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name('X').outputs[0]
    sess.run(tf.local_variables_initializer())

    enhanced_low_resolution_samples = sess.run(Y, feed_dict={X: low_resolution_samples})
    enhanced_low_resolution_samples = np.array(enhanced_low_resolution_samples).astype(np.float32)
    enhanced_low_resolution_samples = enhanced_low_resolution_samples.transpose(0,3,1,2)
    print(enhanced_low_resolution_samples.shape)


np.save('./data/enhanced_GM12878_replicate_down16_chr19_22.npy', enhanced_low_resolution_samples)