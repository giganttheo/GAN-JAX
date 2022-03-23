import jax
import flax
import jax.numpy as jnp
from data.mnist import get_data
from utils import plot, plot_conditional, sample_latent


def train_vanilla():
    key = jax.random.PRNGKey(seed=41)
    data_gen, batches_in_epoch = get_data()
    from models.vanilla_gan import VanillaGan
    models = VanillaGan()
    models.train(data_gen, batches_in_epoch, key)


def train_wgan():
    data_gen, batches_in_epoch = get_data()
    key = jax.random.PRNGKey(seed=41)
    from models.wgan import WGan
    models = WGan()
    models.train(data_gen, batches_in_epoch, key)


def train_conditional_gan():
    data_gen, batches_in_epoch = get_data()
    key = jax.random.PRNGKey(seed=41)
    from models.conditional_gan import ConditionalGan
    models = ConditionalGan()
    models.train(data_gen, batches_in_epoch, key)

def train_infogan():
    data_gen, batches_in_epoch = get_data()
    key = jax.random.PRNGKey(seed=41)
    from models.infogan import InfoGan
    models = InfoGan()
    models.train(data_gen, batches_in_epoch, key)

def train_cyclegan():
    from data.horse2zebra import get_data
    data_gen, batches_in_epoch = get_data()
    key = jax.random.PRNGKey(seed=41)
    from models.cyclegan import CycleGan
    models = CycleGan()
    models.train(data_gen, batches_in_epoch, key)

def main():
    pass

if __name__ == "main":
    main()