#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# In[2]:


# import warnings
# warnings.filterwarnings('ignore')


# ## Data Preprocessing

# ### Generating images for the Training set

# In[3]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# ### Generating images for the Test set

# In[4]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# ### Creating the Training set

# In[5]:


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 class_mode = 'binary')


# ### Creating the Test set

# In[6]:


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            class_mode = 'binary')


# ## Building the CNN

# ### Initialising the CNN

# In[7]:


cnn = tf.keras.models.Sequential()


# ### Convolution

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))


# ### Pooling

# In[9]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# ### Adding another convolutional layers

# In[10]:


#to improve accuracy
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#to improve accuracy
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# ### Flattening

# In[11]:


cnn.add(tf.keras.layers.Flatten())


# ### Full Connection

# In[12]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Output Layer

# In[13]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Training the CNN

# ### Compiling the CNN

# In[14]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[15]:


cnn.fit(
        x=training_set,
        epochs=25,
        validation_data=test_set,
        validation_steps=len(test_set))


# ## Predicting unseen Images

# In[16]:


import numpy as np
from keras.preprocessing import image


# In[17]:


def predict_img(img_path):
    test_image=image.load_img(img_path,target_size=(64,64,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=cnn.predict(test_image)
    training_set.class_indices
    if result[0][0]==1:
        prediction ='Dog..!!'
    else:
        prediction='Cat..!!'
    print(prediction)


# ![](dataset/single_prediction/cat_or_dog_1.jpg)

# In[18]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_1.jpg')


# ![](dataset/single_prediction/cat_or_dog_2.jpg)

# In[19]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_2.jpg')


# ![](dataset/single_prediction/cat_or_dog_7.jpg)

# In[20]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_7.jpg')


# ![](dataset/single_prediction/cat_or_dog_5.jpg)

# In[21]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_5.jpg')


# ![](dataset/single_prediction/cat_or_dog_4.jpg)

# In[22]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_4.jpg')


# ![](dataset/single_prediction/cat_or_dog_6.jpg)

# In[23]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_6.jpg')


# ![](dataset/single_prediction/cat_or_dog_8.jpg)

# In[24]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_8.jpg')


# ![](dataset/single_prediction/cat_or_dog_10.jpg)

# In[25]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_10.jpg')


# ![](dataset/single_prediction/cat_or_dog_9.jpg)

# In[26]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_9.jpg')


# ![](dataset/single_prediction/cat_or_dog_3.jpg)

# In[28]:


predict_img(img_path='dataset/single_prediction/cat_or_dog_3.jpg')

