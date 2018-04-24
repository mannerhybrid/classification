
# coding: utf-8

# ## Image Classification is Bullshit
# 

# In[1]:


import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import csv

tf.reset_default_graph()


# In[2]:


filepath="C:/Users/tp-roslimnh/Desktop"


# In[3]:


r_folder = os.path.join(filepath, "rabbit")
p_folder = os.path.join(filepath, "pika")
csvfile = os.path.join(filepath, "class.csv")


# In[13]:


with open(csvfile, "r") as f:
    rabbit_list = []
    labels = []
    pika_list= []
    lines = [file.strip('\n').split(',') for file in f][1:]
    for i in lines:
        label = i[1]
        if label == '1':
            data = np.array(Image.open(i[0]).resize((256,256)))
            #data = np.expand_dims(data, axis=0)
            pika_list.append(data)
            labels.append(int(label))
            
        else:
            data = np.array(Image.open(i[0]).resize((256,256)))
            #data = np.expand_dims(data, axis=0)
            rabbit_list.append(data)
            labels.append(int(label))

print(rabbit_list[0].shape)
print(pika_list[0].shape)
rabbit_list = rabbit_list + pika_list
labels=np.array(labels)
images = np.array(rabbit_list)
print(images.shape)
print(labels)


# In[5]:


T=tf.one_hot(labels, 2)


with tf.Session() as sess:
    T = sess.run(T)
    print(T.shape)


# In[27]:


x = tf.placeholder(dtype=tf.float32, 
                  shape=(None,256,256,3),
                  name="x")
print(x)
W = tf.truncated_normal(
    shape=[9,9,3,32],
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)
# Convolutional Layer #1
conv1 = tf.layers.conv2d(
  inputs=x,
  filters=32,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
print(pool1)
# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
  inputs=pool1,
  filters=64,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
print(pool2)
# Dense Layer
flat = tf.reshape(pool2, [-1, 64 * 64 * 64])
print(flat)

fcn1 = tf.layers.dense(flat, units=1024)
print(fcn1)
dropout = tf.layers.dropout(
    inputs=fcn1,
    rate=0.4)
logits = tf.layers.dense(inputs=dropout, units=2)
print(logits)

predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

loss = tf.losses.sparse_softmax_cross_entropy(labels=T, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
        loss=loss)
print(train_op)


# In[26]:


with tf.Session() as sess:
    num_epochs=10
    i=0
    tf.initialize_all_variables().run()
    for i in range(10):
        print("Epoch", str(i + 1),"of", str(num_epochs),"...")
        loss, _ =sess.run([loss, train_op], feed_dict={x:images})
        print("Loss:", loss)

