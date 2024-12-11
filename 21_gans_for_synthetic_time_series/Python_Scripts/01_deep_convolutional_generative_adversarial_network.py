#!/usr/bin/env python
# coding: utf-8

# <font size="+3">Deep Convolutional Generative Adversarial Network (DCGAN)</font>

# To illustrate the implementation of a GAN using Python, we use the [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434) (DCGAN) example discussed in the Section 'Evolution of GAN Architectures' to synthesize images from the fashion MNIST dataset. 

# Adapted from the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/generative/dcgan) on generative modeling with the DCGAN architecture.

# # Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (Dense, Reshape, Flatten, Dropout,
                                     BatchNormalization, LeakyReLU,
                                     Conv2D, Conv2DTranspose)
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from IPython import display


# In[3]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')


# In[4]:


dcgan_path = Path('dcgan')
img_path = dcgan_path / 'synthetic_images'
if not img_path.exists():
    img_path.mkdir(parents=True)


# # Sample Image Saver

# In[5]:


def generate_and_save_images(model, epoch, test_input):
    # Training set to false so that every layer runs in inference mode
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5, 5.2))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.suptitle(f'Epoch {epoch:03d}', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.93)
    fig.savefig(img_path / f'epoch_{epoch:03d}.png', dpi=300)
    plt.show()


# # Load and Prepare Data

# ## Get training images

# In[6]:


# use only train images
(train_images, train_labels), (_, _) = fashion_mnist.load_data()


# ## Extract image dimensions

# In[7]:


# get dimensionality
n_images = train_images.shape[0]
h = w = 28


# ## Reshape to 4D input format

# In[8]:


train_images = (train_images
                .reshape(n_images, h, w, 1)
                .astype('float32'))


# ## Normalize pixel values

# In[9]:


# Normalize the images in between -1 and 1
train_images = (train_images - 127.5) / 127.5


# ## Create `tf.data.Dataset`

# In[10]:


BUFFER_SIZE = n_images
BATCH_SIZE = 256


# In[11]:


train_set = (tf.data.Dataset
             .from_tensor_slices(train_images)
             .shuffle(BUFFER_SIZE)
             .batch(BATCH_SIZE))


# In[12]:


train_set


# # Create DCGAN Architecture

# ## Build Generator

# In[13]:


def build_generator():
    return Sequential([Dense(7 * 7 * 256, 
                             use_bias=False,
                             input_shape=(100,), 
                             name='IN'),
                       BatchNormalization(name='BN1'),
                       LeakyReLU(name='RELU1'),
                       Reshape((7, 7, 256), name='SHAPE1'),
                       Conv2DTranspose(128, (5, 5), 
                                       strides=(1, 1),
                                       padding='same', 
                                       use_bias=False,
                                       name='CONV1'),
                       BatchNormalization(name='BN2'),
                       LeakyReLU(name='RELU2'),
                       Conv2DTranspose(64, (5, 5), 
                                       strides=(2, 2),
                                       padding='same', use_bias=False,
                                       name='CONV2'),
                       BatchNormalization(name='BN3'),
                       LeakyReLU(name='RELU3'),
                       Conv2DTranspose(1, (5, 5), 
                                       strides=(2, 2),
                                       padding='same', 
                                       use_bias=False,
                                       activation='tanh', 
                                       name='CONV3')],
                      name='Generator')


# In[14]:


generator = build_generator()


# In[15]:


generator.summary()


# In[16]:


plot_model(generator,
           show_shapes=True,
           dpi=300,
           to_file=(dcgan_path / 'generator.png').as_posix())


# ### Visualize initial image quality

# In[17]:


noise = tf.random.normal([1, 100])
generated_img = generator(noise, training=False)

plt.imshow(generated_img[0, :, :, 0], cmap='gray')
plt.axis('off');


# ## Build Discriminator

# Both the discriminator and generator use a deep CNN architecture, wrapped in a function:

# In[18]:


def build_discriminator():
    return Sequential([Conv2D(64, (5, 5), 
                              strides=(2, 2), 
                              padding='same',
                              input_shape=[28, 28, 1], 
                              name='CONV1'),
                       LeakyReLU(name='RELU1'),
                       Dropout(0.3, name='DO1'),
                       Conv2D(128, (5, 5), 
                              strides=(2, 2),
                              padding='same', 
                              name='CONV2'),
                       LeakyReLU(name='RELU2'),
                       Dropout(0.3, name='DO2'),
                       Flatten(name='FLAT'),
                       Dense(1, name='OUT')],
                      name='Discriminator')


# In[19]:


discriminator = build_discriminator()


# In[20]:


discriminator.summary()


# In[21]:


plot_model(discriminator,
           show_shapes=True,
           dpi=300,
           to_file=(dcgan_path / 'discriminator.png').as_posix())


# ### Show discriminator decision output

# In[22]:


discriminator(generated_img).numpy()


# # Adversarial Training

# In[23]:


cross_entropy = BinaryCrossentropy(from_logits=True)


# ## Generator Loss

# In[24]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# ## Discriminator Loss

# In[25]:


def discriminator_loss(true_output, fake_output):
    true_loss = cross_entropy(tf.ones_like(true_output), true_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return true_loss + fake_loss


# ## Optimizers

# In[26]:


gen_optimizer = Adam(1e-4)
dis_optimizer = Adam(1e-4)


# ## Checkpoints

# In[27]:


checkpoints_dir = dcgan_path / 'training_chpk'
checkpoint_prefix = checkpoints_dir / 'ckpt'
checkpoint = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                                 dis_optimizer=dis_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Training Parameters

# In[28]:


EPOCHS = 100
noise_dim = 100


# In[29]:


# for gif generation
num_ex_to_gen = 16
seed = tf.random.normal([num_ex_to_gen, noise_dim])


# The training begins with generator given a random seed as an input which is used to generate an image. The role of discriminator is to compare true image from training set with the fake image from generator. The loss is then calculated for both the models and weights are then updated through gradient descent. As we proceed further we will be having an efficient generator that is able to generate fake images close to the true images and an accurate discriminator that is able to accurately distinguish fake images from true images.

# ## Training Step

# In[31]:


@tf.function
def train_step(images):
    # generate the random input for the generator
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        # get the generator output
        generated_img = generator(noise, training=True)
        
        # collect discriminator decisions regarding real and fake input
        true_output = discriminator(images, training=True)
        fake_output = discriminator(generated_img, training=True)
        
        # compute the loss for each model
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(true_output, fake_output)
        
    # compute the gradients for each loss with respect to the model variables
    grad_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # apply the gradient to complete the backpropagation step
    gen_optimizer.apply_gradients(zip(grad_generator, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(grad_discriminator, discriminator.trainable_variables))


# ## Training Loop

# In[32]:


def train(dataset, epochs, save_every=10):
    for epoch in tqdm(range(epochs)):
        start = time()

        for img_batch in dataset:
            train_step(img_batch)

        # produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 10 EPOCHS
        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Generator after final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


# In[33]:


train(train_set, EPOCHS)


# ## Restore last checkpoint

# In[34]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir));


# # Generate animated GIF

# In[35]:


out_file = dcgan_path / 'fashion_mnist.gif'


# In[36]:


with imageio.get_writer(out_file, mode='I') as writer:
    filenames = sorted(list(img_path.glob('*.png')))
    
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


# In[37]:


display.Image(filename=out_file)

