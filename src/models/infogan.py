import jax
import jax.numpy as jnp
from flax import linen as nn
from utils import sample_latent_categorical as sample_latent
from architecture.dcgan import Generator
from architecture.dcgan import DiscriminatorAndRecognitionNetwork as Discriminator

#Losses

def bce_logits(logit, label):
  """
  Implements the BCE with logits loss, as described:
  https://github.com/pytorch/pytorch/issues/751
  """
  neg_abs = -jnp.abs(logit)
  batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
  return jnp.mean(batch_bce)


def loss_mutual_information(code_cat, q_cat):
  cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
  mi_loss = cat_loss
  return mi_loss


def loss_generator(params_g, params_d, vars_g, vars_d, data, key):
  batch_size = data.shape[0]
  latent, code_cat = sample_latent(key, (batch_size, 64), (batch_size, ))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  (preds, q), vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  # Calculate Mutual Information loss
  q_cat = nn.log_softmax(q, axis=-1)
  loss_mi = loss_mutual_information(code_cat, q_cat)

  # Calculate Discriminator loss
  loss_d = -jnp.mean(jnp.log(nn.sigmoid(preds)))

  loss = loss_d + loss_mi
  return loss, (vars_g, vars_d)


def loss_discriminator(params_d, params_g, vars_g, vars_d, data, key):
  batch_size = data.shape[0]
  latent, code_cat = sample_latent(key, (batch_size, 64), (batch_size, ))

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      latent, mutable=['batch_stats']
  )

  (fake_preds, q), vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )
  (real_preds, _), vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      data, mutable=['batch_stats']
  )

  # Calculate Mutual Information loss
  q_cat = nn.log_softmax(q, axis=-1)
  loss_mi = loss_mutual_information(code_cat, q_cat)

  real_loss = bce_logits(real_preds, jnp.ones((batch_size,), dtype=jnp.int32))
  fake_loss = bce_logits(fake_preds, jnp.zeros((batch_size,), dtype=jnp.int32))

  loss = (real_loss + fake_loss) / 2 + loss_mi

  return loss, (vars_g, vars_d)


#Train and eval functions


@jax.jit
def train_step(data, vars_g, vars_d, optim_g, optim_d, key):
  key, key_gen, key_disc = jax.random.split(key, 3)

  # Train the discriminator
  grad_fn_discriminator = jax.value_and_grad(loss_discriminator, has_aux=True)
  (loss_d, (vars_g, vars_d)), grad_d = grad_fn_discriminator(
      optim_d.target, optim_g.target, vars_g, vars_d, data, key_disc
  )
  optim_d = optim_d.apply_gradient(grad_d)

  # Train the generator
  grad_fn_generator = jax.value_and_grad(loss_generator, has_aux=True)
  (loss_g, (vars_g, vars_d)), grad_g = grad_fn_generator(
      optim_g.target, optim_d.target, vars_g, vars_d, data, key_gen
  )

  optim_g = optim_g.apply_gradient(grad_g)

  loss = {'generator': loss_g, 'discriminator': loss_d}

  return loss, vars_g, vars_d, optim_g, optim_d, key


@jax.jit
def eval_step(params, vars, latent):  
  fake_data, _ = Generator(training=False).apply(
      {'params': params, 'batch_stats': vars['batch_stats']},
      latent, mutable=['batch_stats']
  )

  return fake_data