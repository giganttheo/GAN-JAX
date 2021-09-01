import jax
import jax.numpy as jnp
from flax import linen as nn
from utils import sample_latent
from architecture.DCGAN import Generator, Discriminator

#Losses

def bce_logits(input, target):
  """
  Implements the BCE with logits loss, as described:
  https://github.com/pytorch/pytorch/issues/751
  """
  neg_abs = -jnp.abs(input)
  batch_bce = jnp.maximum(input, 0) - input * target + jnp.log(1 + jnp.exp(neg_abs))
  return jnp.mean(batch_bce)


def loss_generator(params_g, params_d, vars_g, vars_d, data, key):
  latent = sample_latent(key, shape=(data.shape[0], 64))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  fake_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  loss = -jnp.mean(jnp.log(nn.sigmoid(fake_preds)))
  return loss, (vars_g, vars_d)


def loss_discriminator(params_d, params_g, vars_g, vars_d, data, key):
  latent = sample_latent(key, shape=(data.shape[0], 64))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  fake_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  real_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      data, mutable=['batch_stats']
  )

  real_loss = bce_logits(real_preds, jnp.ones((data.shape[0],), dtype=jnp.int32))
  fake_loss = bce_logits(fake_preds, jnp.zeros((data.shape[0],), dtype=jnp.int32))
  loss = (real_loss + fake_loss) / 2

  return loss, (vars_g, vars_d)


#Train and eval functions

@jax.jit
def train_step(data, vars_g, vars_d, optim_g, optim_d, rng):
  key, key_gen, key_disc = jax.random.split(rng, 3)

  # Train the generator
  grad_fn_generator = jax.value_and_grad(loss_generator, has_aux=True)
  (loss_g, (vars_g, vars_d)), grad_g = grad_fn_generator(
      optim_g.target, optim_d.target, vars_g, vars_d, data, key_gen
  )

  optim_g = optim_g.apply_gradient(grad_g)
  
  # Train the discriminator
  grad_fn_discriminator = jax.value_and_grad(loss_discriminator, has_aux=True)
  (loss_d, (vars_g, vars_d)), grad_d = grad_fn_discriminator(
      optim_d.target, optim_g.target, vars_g, vars_d, data, key_disc
  )

  optim_d = optim_d.apply_gradient(grad_d)

  loss = {'generator': loss_g, 'discriminator': loss_d}
  return loss, vars_g, vars_d, optim_g, optim_d, key


@jax.jit
def eval_step(params, vars, latent):  
  fake_data, _ = Generator(training=False).apply(
      {'params': params, 'batch_stats': vars['batch_stats']},
      latent, mutable=['batch_stats']
  )

  return fake_data
