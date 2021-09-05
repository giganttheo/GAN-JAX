import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from utils import sample_latent
from architecture.dcgan import Generator, Critic
from utils import plot, sample_latent
from models.base_model import Model

#Losses

@jax.jit
def loss_generator(params_g, params_c, vars_g, vars_c, data, key):
  latent = sample_latent(key, shape=(data.shape[0], 64))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  fake_value, vars_c = Critic().apply(
      {'params': params_c, 'batch_stats': vars_c['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  loss = jnp.mean(-fake_value)
  return loss, (vars_g, vars_c)


@jax.partial(jax.vmap, in_axes=(None, None, 0))
@jax.partial(jax.grad, argnums=2)
def critic_forward(params, vars, input_image):
  """Helper function to calculate the gradients with respect to the input."""
  value, _ = Critic().apply(
      {'params': params, 'batch_stats': vars['batch_stats']},
      input_image, mutable=['batch_stats']
  )
  return value[0, 0]


@jax.jit
def loss_critic(params_c, params_g, vars_g, vars_c, data, key):
  latent = sample_latent(key, shape=(data.shape[0], 64))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  fake_value, vars_c = Critic().apply(
      {'params': params_c, 'batch_stats': vars_c['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  real_value, vars_c = Critic().apply(
      {'params': params_c, 'batch_stats': vars_c['batch_stats']},
      data, mutable=['batch_stats']
  )

  # Interpolate between fake and real images with epsilon
  epsilon = jax.random.uniform(key, shape=(data.shape[0], 1, 1, 1))
  data_mix = data * epsilon + fake_data * (1 - epsilon)

  # Fetch the gradient penalty
  gradients = critic_forward(params_c, vars_c, data_mix)
  gradients    = gradients.reshape((gradients.shape[0], -1))
  grad_norm    = jnp.linalg.norm(gradients, axis=1)
  grad_penalty = ((grad_norm - 1) ** 2).mean()

  # here we use 10 as a fixed parameter as a cost of the penalty.
  loss = - real_value.mean() + fake_value.mean() + 10 * grad_penalty

  return loss, (vars_g, vars_c)


#Train and eval functions

@jax.jit
def train_step(data, vars_g, vars_c, optim_g, optim_c, rng):
  key, key_gen, key_crit = jax.random.split(rng, 3)

  # Train the generator
  grad_fn_generator = jax.value_and_grad(loss_generator, has_aux=True)
  (loss_g, (vars_g, vars_c)), grad_g = grad_fn_generator(
      optim_g.target, optim_c.target, vars_g, vars_c, data, key_gen
  )

  optim_g = optim_g.apply_gradient(grad_g)
  
  # We train the critic iteratively, ensuring the penalty has sufficient effect.
  grad_fn_critic = jax.value_and_grad(loss_critic, has_aux=True)
  for _ in range(5):
    key, key_crit = jax.random.split(rng, 2)
    (loss_c, (vars_g, vars_c)), grad_c = grad_fn_critic(
        optim_c.target, optim_g.target, vars_g, vars_c, data, key_crit
    )

    optim_c = optim_c.apply_gradient(grad_c)
  
  loss = {'generator': loss_g, 'critic': loss_c}
  return loss, vars_g, vars_c, optim_g, optim_c, key


@jax.jit
def eval_step(params, vars, latent):  
  fake_data, _ = Generator(training=False).apply(
      {'params': params, 'batch_stats': vars['batch_stats']},
      latent, mutable=['batch_stats']
  )

  return fake_data



#Training loop

class Wgan(Model):
    def __init__(self):
        pass

    def train(self, data_gen, batches_in_epoch, key, verbose=1):
        key, key_gen, key_crit, key_latent = jax.random.split(key, 4)

        # Retrieve shapes for generator and discriminator input.
        latent = sample_latent(key_latent, shape=(100, 64)) #TODO shape
        image_shape = next(data_gen).shape

        # Generate initial variables (parameters and batch statistics).
        vars_g = Generator().init(key_gen, jnp.ones(latent.shape, jnp.float32))
        vars_c = Critic().init(key_crit, jnp.ones(image_shape, jnp.float32))

        # Create optimizers.
        optim_g = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_g['params']) #TODO hp
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
            if verbose:
                plot(sample, loss, epoch)

    def eval(self):
        pass