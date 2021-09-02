import matplotlib.pyplot as plt
import jax
from IPython.display import clear_output


def plot(images, loss, epoch):
  clear_output(True)

  # First plot the losses.
  fig, ax = plt.subplots(figsize=(10, 4))
  for key in loss.keys():
      ax.plot(loss[key], label=f'{key} loss')
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  fig.suptitle(f"Epoch {epoch}")

  # Next, plot the static samples.
  fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), tight_layout=bool)
  for ax, image in zip(sum(axes.tolist(), []), images):
    ax.imshow(image[:, :, 0], cmap='gray')
    ax.set_axis_off()

  plt.show()


def plot_conditional(images, loss, labels, epoch):
  clear_output(True)

  # First plot the losses.
  fig, ax = plt.subplots(figsize=(10, 4))
  for key in loss.keys():
      ax.plot(loss[key], label=f'{key} loss')
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  fig.suptitle(f"Epoch {epoch}")


  fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), tight_layout=True)
  for ax, image, label in zip(sum(axes.tolist(), []), images, labels):
    ax.imshow(image[:, :, 0], cmap='gray')
    ax.set_axis_off()
    ax.text(0, 0, f"{label}", bbox=dict(facecolor='white'))

  plt.show()

def sample_latent(key, shape):
  return jax.random.normal(key, shape=shape)

def sample_latent_categorical(key, shape_noise, shape_cat):
  noise_key, cat_key = jax.random.split(key, 2)
  
  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, 10)
  code_cat = jax.nn.one_hot(code_cat, 10)

  latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat

@jax.jit
def fetch_oh_labels(labels, num_classes=10):
  oh_labels = jax.nn.one_hot(labels, num_classes=num_classes)
  oh_labels_img = oh_labels[:, None, None, :].repeat(28, 1).repeat(28, 2)
  return oh_labels, oh_labels_img