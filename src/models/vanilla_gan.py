import jax
#TODO import

from losses import loss_generator, loss_discriminator
from architecture.conv_net import Generator

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
