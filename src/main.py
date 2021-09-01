import jax
import flax
import jax.numpy as jnp
from models.vanilla_gan import train_step, eval_step
from architecture.dcgan import Generator, Critic, Discriminator
from data.mnist import get_data
from utils import plot, sample_latent


def train_vanilla():
    key = jax.random.PRNGKey(seed=41)
    key, key_gen, key_disc, key_latent = jax.random.split(key, 4)

    data_gen, batches_in_epoch = get_data()

    # Retrieve shapes for generator and discriminator input.
    latent = sample_latent(key_latent, shape=(100, 64))
    image_shape = next(data_gen).shape

    # Generate initial variables (parameters and batch statistics).
    vars_g = Generator().init(key_gen, jnp.ones(latent.shape, jnp.float32))
    vars_d = Discriminator().init(key_disc, jnp.ones(image_shape, jnp.float32))

    # Create optimizers.
    optim_g = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_g['params'])
    optim_d = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_d['params'])

    loss = {'generator': [], 'discriminator': []}

    for epoch in range(1, 51):
        for batch in range(batches_in_epoch):
            data = next(data_gen)

            batch_loss, vars_g, vars_d, optim_g, optim_d, key = train_step(
                data, vars_g, vars_d, optim_g, optim_d, key
            )

            loss['generator'].append(batch_loss['generator'])
            loss['discriminator'].append(batch_loss['discriminator'])
    
    sample = eval_step(optim_g.target, vars_g, latent)
    plot(sample, loss, epoch)

def train_wgan():
    data_gen, batches_in_epoch = get_data()

    key = jax.random.PRNGKey(seed=41)
    key, key_gen, key_crit, key_latent = jax.random.split(key, 4)

    # Retrieve shapes for generator and discriminator input.
    latent = sample_latent(key_latent, shape=(100, 64))
    image_shape = next(data_gen).shape

    # Generate initial variables (parameters and batch statistics).
    vars_g = Generator().init(key_gen, jnp.ones(latent.shape, jnp.float32))
    vars_c = Critic().init(key_crit, jnp.ones(image_shape, jnp.float32))

    # Create optimizers.
    optim_g = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_g['params'])
    optim_c = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_c['params'])

    loss = {'generator': [], 'critic': []}

    for epoch in range(1, 51):
      for batch in range(batches_in_epoch):
        data = next(data_gen)

        batch_loss, vars_g, vars_c, optim_g, optim_c, key = train_step(
            data, vars_g, vars_c, optim_g, optim_c, key
        )

        loss['generator'].append(batch_loss['generator'])
        loss['critic'].append(batch_loss['critic'])
      
      sample = eval_step(optim_g.target, vars_g, latent)
      plot(sample, loss, epoch)


def main():
    pass

if __name__ == "main":
    main()