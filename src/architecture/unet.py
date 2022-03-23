from functools import partial
from typing import Any, Callable, Sequence, Tuple
from jax.nn.initializers import normal as normal_init
from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any

# U-Net architecture, code adapted from\
# https://github.com/tensorflow/examples/blob/79d3c093f0f396305e6bcf12ea4d5d85e07275d0/tensorflow_examples/models/pix2pix/pix2pix.py

class DownSampleBlock(nn.Module):
  filters: int
  size: Tuple[int, int]
  conv: ModuleDef
  norm: ModuleDef
  
  @nn.compact
  def __call__(self, x):
    y = self.conv(self.filters, self.size, strides=2, padding='same')(x)
    y = self.norm()(y)
    y = nn.leaky_relu(y, 0.3)
    return y

class UpSampleBlock(nn.Module):
  filters: int
  size: Tuple[int, int]
  conv_transpose: ModuleDef
  norm: ModuleDef
  use_dropout: bool = False
  
  @nn.compact
  def __call__(self, x):
    y = self.conv_transpose(self.filters, self.size, strides=(2,2), padding='SAME')(x)
    y = self.norm()(y)
    y = nn.relu(y)
    if self.use_dropout :
      y = nn.Dropout(0.5)(y)
    return y

class UNetGenerator(nn.Module):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004)"""
  stage_sizes_down: Sequence[int]
  stage_sizes_up: Sequence[int]
  output_channels: int = 3
  downsample: ModuleDef = DownSampleBlock
  upsample: ModuleDef = UpSampleBlock
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool=True):

    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    conv_t = partial(nn.ConvTranspose, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    skips = []
    for down_size in self.stage_sizes_down :
      x = self.downsample(down_size, (4,4), conv, norm)(x)
      skips.append(x)
    skips = reversed(skips[:-1])
    for up_size, skip in zip(self.stage_sizes_up, skips):
      x = self.upsample(up_size, (4,4), conv_t, norm)(x)
      x = jnp.concatenate([x, skip], axis=-1)
    x = self.upsample(self.output_channels, (4,4), conv_t, norm)(x)
    return x

class PatchGanDiscriminator(nn.Module):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  """
  downsample: ModuleDef = DownSampleBlock
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool=True):

    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)
    padding2D = partial(jnp.pad, pad_width=[(0,0), (1,1), (1,1), (0,0)])
    # (bs, 256, 256, channels)
    for down_size in [64, 128, 256] :
      x = self.downsample(down_size, (4,4), conv, norm)(x)
    # (bs, 32, 32, 256)
    x = padding2D(x) # (bs, 34, 34, 256)
    x = conv(512, (4,4), strides=1, padding='valid')(x) # (bs, 31, 31, 512)
    x = norm()(x)
    x = nn.leaky_relu(x, 0.3)
    x = padding2D(x) # (bs, 33, 33, 512)
    x = conv(1, (4,4), strides=1, padding="valid")(x) #(bs, 30, 30, 1)
    return x

UNetDefault = partial(UNetGenerator,
             stage_sizes_up=[64, 128, 256],
             stage_sizes_down=[256, 128, 64],
             output_channels=3)

UNetPix2pix = partial(UNetGenerator,
             stage_sizes_up=[64, 128, 256, 512, 512, 512, 512, 512],
             stage_sizes_down=[512, 256, 128, 64],
             output_channels=3)