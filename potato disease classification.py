#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers


# In[3]:


df = tf.keras.preprocessing.image_dataset_from_directory("D:\deeplearn",
                                                        shuffle=True ,
                                                        image_size = (256,256),
                                                        batch_size = 32)


# In[4]:


classes =df.class_names
classes


# In[5]:


len(df)


# In[6]:


for image_batch,label_batch in df.take(1):
    #print(image_batch[0].numpy())
    #print(image_batch[0].shape)#channels = 3 (RGB)
    print(image_batch.shape)
    print(label_batch.numpy())


# In[7]:


plt.figure(figsize=(10,10))
for image_batch,label_batch in df.take(1):
    for i in range(12):
     ax = plt.subplot(3,4,i+1)
     plt.title(classes[label_batch[i]])
     plt.imshow(image_batch[i].numpy().astype('uint8'))
     plt.axis('off')


# In[9]:


train_size = 0.8
len(df)*train_size


# In[10]:


train_ds = df.take(54)
len(train_ds)


# In[11]:


test_ds = df.skip(54)
len(test_ds)


# In[12]:


val_size=0.1
len(df)*val_size


# In[13]:


val_ds = test_ds.take(6)
len(val_ds)


# In[14]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[38]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[25]:


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50


# In[26]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(df)


# In[27]:


#cache shuffle prefetech


# In[28]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[29]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(256,256),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# In[30]:


#data augmentation
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])


# In[31]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[32]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[33]:


model.summary()


# In[34]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
# here i am using adam Optimizer, SparseCategoricalCrossentropy for losses, accuracy as a metric


# In[35]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)


# In[36]:


scores = model.evaluate(test_ds)


# In[37]:


scores


# In[39]:


history.history['accuracy']


# In[41]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",classes[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",classes[np.argmax(batch_prediction[0])])


# In[ ]:




