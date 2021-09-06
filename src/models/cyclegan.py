import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from utils import sample_latent, plot
from architecture.resnet import ResNet18 as Generator
from architecture.resnet import Discriminator
from models.base_model import Model

#Losses

def bce_logits(input, target):
  """
  Implements the BCE with logits loss, as described:
  https://github.com/pytorch/pytorch/issues/751
  """
  neg_abs = -jnp.abs(input)
  batch_bce = jnp.maximum(input, 0) - input * target + jnp.log(1 + jnp.exp(neg_abs))
  return jnp.mean(batch_bce)


def loss_generator(params_g, params_d, vars_g, vars_d, data):

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      data, mutable=['batch_stats']
  )

  fake_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  loss = bce_logits(jnp.sigmoid(fake_preds), jnp.ones((data.shape[0],), dtype=jnp.int32))
  return loss, (vars_g, vars_d)


def loss_discriminator(params_d, params_g, vars_g, vars_d, data):

  fake_data, vars_g = Generator().apply(
      {'params': params_g, 'batch_stats': vars_g['batch_stats']},
      data, mutable=['batch_stats']
  )

  fake_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      fake_data, mutable=['batch_stats']
  )

  real_preds, vars_d = Discriminator().apply(
      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
      data, mutable=['batch_stats']
  )

  real_loss = bce_logits(jnp.sigmoid(real_preds), jnp.ones((data.shape[0],), dtype=jnp.int32))
  fake_loss = bce_logits(jnp.sigmoid(fake_preds), jnp.zeros((data.shape[0],), dtype=jnp.int32))
  loss = (real_loss + fake_loss) / 2

  return loss, (vars_g, vars_d)


def loss_cycle(params_g_1, params_g_2, vars_g_1, vars_g_2, data):
    fake_data_2, vars_g_1 = Generator().apply(
        {'params': params_g_1, 'batch_stats': vars_g_1['batch_stats']},
        data, mutable=['batch_stats']
    )
    fake_data_1, vars_g_2 = Generator().apply(
        {'params': params_g_2, 'batch_stats': vars_g_2['batch_stats']},
        fake_data_2, mutable=['batch_stats']
    )

    cycle_loss = jnp.mean(jnp.abs(data - fake_data_1))

    return cycle_loss, (vars_g_1, vars_g_2)

def loss_identity(params_g, vars_g, data):
    same_data, vars_g = Generator().apply(
        {'params': params_g, 'batch_stats': vars_g['batch_stats']},
        data, mutable=['batch_stats']
    )
    identity_loss = jnp.mean(jnp.abs(data - same_data))

    return identity_loss, vars_g

def loss_total_gen(params_g_1, params_g_2, params_d_1, params_d_2, vars_g_1, vars_g_2, vars_d_1, vars_d_2, data_1, data_2):
    gen_1_loss, (vars_g_1, vars_d_1) = loss_generator(params_g_1, params_d_1, vars_g_1, vars_d_1, data_1)
    cycle_1_loss, (vars_g_1, vars_g_2) = loss_cycle(params_g_1, params_g_2, vars_g_1, vars_g_2, data_1)
    cycle_2_loss, (vars_g_2, vars_g_1) = loss_cycle(params_g_2, params_g_1, vars_g_2, vars_g_1, data_2)
    identity_1_loss, vars_g_1 = loss_identity(params_g_1, vars_g_1, data_1)

    total_loss = gen_1_loss + cycle_1_loss + cycle_2_loss + identity_1_loss #TODO add lambdas
    return total_loss, (vars_g_1, vars_g_2, vars_d_1, vars_d_2)



  
#Train and eval functions

@jax.jit
def train_step(data_A, data_B, vars_g_A, vars_d_A, vars_g_B, vars_d_B, optim_g_A, optim_d_A, optim_g_B, optim_d_B):

    # Train the generators

    #generator A
    grad_fn_generator_A = jax.value_and_grad(loss_total_gen, has_aux=True)
    (loss_g_A, (vars_g_A, vars_g_B, vars_d_A, vars_d_B)), grad_g_A = grad_fn_generator_A(
        optim_g_A.target, optim_g_B.target, optim_d_A.target, optim_d_B.target, vars_g_A, vars_g_B, vars_d_A, vars_d_B, data_A, data_B
    )

    optim_g_A = optim_g_A.apply_gradient(grad_g_A)

    #generator B
    grad_fn_generator_B = jax.value_and_grad(loss_generator, has_aux=True)
    (loss_g_B, (vars_g_B, vars_g_A, vars_d_B, vars_d_A)), grad_g_B = grad_fn_generator_B(
        optim_g_B.target, optim_g_A.target, optim_d_B.target, optim_d_A.target, vars_g_B, vars_g_A, vars_d_B, vars_d_A, data_B, data_A
    )

    optim_g_B = optim_g_B.apply_gradient(grad_g_B)

    # Train the discriminator

    #discriminator A
    grad_fn_discriminator_A = jax.value_and_grad(loss_discriminator, has_aux=True)
    (loss_d_A, (vars_g_A, vars_d_A)), grad_d_A = grad_fn_discriminator_A(
        optim_d_A.target, optim_g_A.target, vars_g_A, vars_d_A, data_A
    )

    optim_d_A = optim_d_A.apply_gradient(grad_d_A)

    #discriminator B
    grad_fn_discriminator_B = jax.value_and_grad(loss_discriminator, has_aux=True)
    (loss_d_B, (vars_g_B, vars_d_B)), grad_d_B = grad_fn_discriminator_B(
        optim_d_B.target, optim_g_B.target, vars_g_B, vars_d_B, data_B
    )

    optim_d_B = optim_d_B.apply_gradient(grad_d_B)

    loss = {'generator_A': loss_g_A, 'generator_B': loss_g_B, 'discriminator_A': loss_d_A, 'dicriminator_B': loss_d_B}
    return loss, vars_g_A, vars_d_A, vars_g_B, vars_d_B, optim_g_A, optim_d_B


@jax.jit
def eval_step(params, vars, data):  
  fake_data, _ = Generator(training=False).apply(
      {'params': params, 'batch_stats': vars['batch_stats']},
      data, mutable=['batch_stats']
  )

  return fake_data


#Training loop 

class CycleGan(Model):


    def train(self, data_gen, batches_in_epoch, key, verbose=1):
        epochs = 51
        key, key_gen, key_disc, key_latent = jax.random.split(key, 4)

        data_gen_A, data_gen_B = data_gen[0], data_gen[1]
        batches_in_epoch = batches_in_epoch[0]

        # Retrieve shapes for generator and discriminator input.
        image_shape = next(data_gen_A).shape

        # Generate initial variables (parameters and batch statistics).

        vars_g_A = Generator().init(key_gen, jnp.ones(image_shape, jnp.float32))
        vars_g_B = Generator().init(key_gen, jnp.ones(image_shape, jnp.float32))
        vars_d_A = Discriminator().init(key_disc, jnp.ones(image_shape, jnp.float32))
        vars_d_B = Discriminator().init(key_disc, jnp.ones(image_shape, jnp.float32))

        # Create optimizers.
        optim_g_A = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_g_A['params'])
        optim_g_B = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_g_B['params'])
        optim_d_A = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_d_A['params'])
        optim_d_B = flax.optim.Adam(0.0002, 0.5, 0.999).create(vars_d_B['params'])

        loss = {'generator_A': [], 'discriminator_A': [], 'generator_B': [], 'discriminator_B': []}

        for epoch in range(1, epochs):
            for batch in range(batches_in_epoch):
                data_A = next(data_gen_A)
                data_B = next(data_gen_B)

                batch_loss, vars_g_A, vars_d_A, vars_g_B, vars_d_B, optim_g_A, optim_d_B = train_step(
                    data_A, data_B, vars_g_A, vars_d_A, vars_g_B, vars_d_B, optim_g_A, optim_d_A, optim_g_B, optim_d_B
                    )

                loss['generator_A'].append(batch_loss['generator_A'])
                loss['discriminator_A'].append(batch_loss['discriminator_A'])
                loss['generator_B'].append(batch_loss['generator_B'])
                loss['discriminator_B'].append(batch_loss['discriminator_B'])
        
            sample = eval_step(optim_g_A.target, vars_g_A, data_A)
            plot(sample, loss, epoch)