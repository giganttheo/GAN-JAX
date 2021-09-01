import tensorflow as tf
import tensorflow_datasets as tfds


""" #Unconditionnal gans
def set_range(batch):
  batch = tf.image.convert_image_dtype(batch['image'], tf.float32)
  batch = (batch - 0.5) / 0.5  # tanh range is -1, 1
  return batch
 
def get_data():
    mnist_data = tfds.load("mnist")['train']
    batches_in_epoch = len(mnist_data) // 128
    data_gen = iter(tfds.as_numpy(
        mnist_data
            .map(set_range)
            .cache()
            .shuffle(len(mnist_data), seed=41)
            .repeat()
            .batch(128)
    ))
    return data_gen, batches_in_epoch
"""

#conditionnal gans
def set_range(batch):
  batch, labels = batch['image'], batch['label']  #  We now add the labels to the generator.
  batch = tf.image.convert_image_dtype(batch, tf.float32)
  batch = (batch - 0.5) / 0.5  # tanh range is -1, 1
  return (batch, labels)
 
def get_data():
    mnist_data = tfds.load("mnist")['train']
    batches_in_epoch = len(mnist_data) // 128
    
    data_gen = iter(tfds.as_numpy(
        mnist_data
            .map(set_range)
            .cache()
            .shuffle(len(mnist_data), seed=42)
            .repeat()
            .batch(128)
    ))
    return data_gen, batches_in_epoch