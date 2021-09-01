import matplotlib.pyplot as plt
import jax

def plot(images, loss, epoch):
  clear_output(True)

  # First plot the losses.
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(loss['generator'], label='Generator Loss')
  ax.plot(loss['discriminator'], label='Discriminator Loss')
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  fig.suptitle(f"Epoch {epoch}")

  # Next, plot the static samples.
  fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), tight_layout=bool)
  for ax, image in zip(sum(axes.tolist(), []), images):
    ax.imshow(image[:, :, 0], cmap='gray')
    ax.set_axis_off()

  plt.show()

def sample_latent(key, shape):
  return jax.random.normal(key, shape=shape)