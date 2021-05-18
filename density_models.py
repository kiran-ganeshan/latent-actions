import jax
import numpy as np
import os, sys
from jax import numpy as jnp
from jax import jit
from flax import linen as nn
from flax import optim
from flax.training.train_state import TrainState
from jax import value_and_grad
from jax import random
from jax import lax
from jax.experimental.host_callback import id_print
from tqdm import tqdm
import tensorflow_datasets as tfds
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from flax.linen.initializers import zeros, lecun_normal
from flax.core.frozen_dict import FrozenDict, V
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
from flax.struct import dataclass
import io
import datetime
from typing import Sequence
from jax import partial
from utils import image_grid, get_datasets, one_hot, write_summary
from optax import adam


FLAGS = flags.FLAGS
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
FLAGS(sys.argv)

class MLP(nn.Module):

    model_name : str = 'mlp'
    hidden_sizes : Sequence[int] = (100,)
    input_size : int = 784
    output_size : int = 10

    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, self.input_size))
        return self.mlp(x)

    def mlp(self, x):
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.Dense(features=self.output_size)(x)
        return x


class Decoder(MLP):
    latent_dim : int = 10
    hidden_sizes : Sequence[int] = (100, 256)
    input_size : int = latent_dim
    output_size : int = 784

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        x = nn.log_sigmoid(x).reshape((-1, 28, 28, 1))
        return x


class Encoder(MLP):
    latent_dim : int = 10
    hidden_sizes : Sequence[int] = (256,)
    input_size : int = 784
    output_size : int = 100

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        mu = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        log_sigma = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        return mu, log_sigma

class TrainState(TrainState):
    batch_stats : FrozenDict[str, any]

@dataclass
class VAETrainState:
    enc_state : TrainState
    dec_state : TrainState

@dataclass
class VAELearner():
    model_name : str = 'vae'
    latent_dim : int = 10
    learning_rate : float = 1e-4
    beta1 : float = 0.5
    beta2 : float = 0.9
    beta : float = 0.5

    @partial(jit, static_argnums=0)
    def initial_state(self, rng):
        enc_rng, dec_rng = random.split(rng)
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(latent_dim=self.latent_dim)
        enc_params = encoder.init(enc_rng, jnp.ones((10, 28, 28, 1), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim), jnp.float32))
        enc_optim = adam(self.learning_rate, self.beta1, self.beta2)
        dec_optim = adam(self.learning_rate, self.beta1, self.beta2)
        enc_state = TrainState.create(apply_fn=encoder.apply, params=enc_params, tx=enc_optim, batch_stats={})
        dec_state = TrainState.create(apply_fn=decoder.apply, params=dec_params, tx=dec_optim, batch_stats={})
        train_state = VAETrainState(enc_state=enc_state, dec_state=dec_state)
        return train_state
        
    def compute_loss(self, enc_apply, dec_apply, enc_params, dec_params, rng, inputs, labels):
        mu, log_sigma = enc_apply(enc_params, inputs)
        r = random.normal(rng, (mu.shape[0], self.latent_dim))
        codes = mu + r * jnp.exp(log_sigma)
        reconst = dec_apply(dec_params, codes)
        def kl_loss(mu, log_sigma):
            kl_exp = 1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma)
            return - self.beta * jnp.mean(jnp.sum(kl_exp, axis=-1))
        def bce_loss(inputs, outputs):
            bce_exp = inputs * outputs + (1 - inputs) * jnp.log(-jnp.expm1(outputs) + 1e-9)
            return -jnp.mean(jnp.sum(bce_exp, axis=(1, 2, 3)))
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), kl_loss(mu, log_sigma), aux

    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, inputs, labels):
        def loss_fn(enc_params, dec_params, rng):
            reconst_loss, kl_penalty, aux = self.compute_loss(train_state.enc_state.apply_fn,
                                                              train_state.dec_state.apply_fn,
                                                              enc_params, dec_params, rng, inputs, labels)
            metrics = {'loss': reconst_loss + kl_penalty, 
                       'reconst_loss': reconst_loss, 
                       'kl_loss': kl_penalty}
            return reconst_loss + kl_penalty, metrics
        
        step_rng, rng = random.split(rng)
        enc_params, dec_params = train_state.enc_state.params, train_state.dec_state.params
        val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
        (loss, batch_stats), (enc_grads, dec_grads) = val_and_grad(enc_params, dec_params, step_rng)

        new_enc_state = train_state.enc_state.apply_gradients(grads=enc_grads)
        new_dec_state = train_state.dec_state.apply_gradients(grads=dec_grads)
        new_train_state = VAETrainState(enc_state=new_enc_state, dec_state=new_dec_state)

        return new_train_state, rng, batch_stats

    @partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        codes = random.normal(rng, (25, self.latent_dim))
        dec_params = train_state.dec_state.params
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        return reconst

    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, inputs, labels):
        loss_rng, generate_rng = random.split(rng)
        reconst_loss, kl_penalty, aux = self.compute_loss(train_state.enc_state.apply_fn,
                                                          train_state.dec_state.apply_fn,
                                                          train_state.enc_state.params, 
                                                          train_state.dec_state.params,
                                                          loss_rng, inputs, labels)
        generated = self.generate(train_state, generate_rng)
        metrics = {'loss': reconst_loss + kl_penalty, 
                   'reconst_loss': reconst_loss, 
                   'kl_loss': kl_penalty, **aux,
                   'generated': jnp.exp(generated)}
        return metrics
        

epochs = 20
batch_size = 60
test_interval = 10
train, test = get_datasets()
coder = VAELearner(learning_rate=3e-4,
                   beta1=0.5,
                   beta2=0.9,
                   latent_dim=10)
rng, init_rng = random.split(random.PRNGKey(0))
state = coder.initial_state(init_rng)

path = os.path.join("./tmp/", coder.model_name)
path = os.path.join(path, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
summary_writer = SummaryWriter(path)

steps_per_epoch = train['image'].shape[0] // batch_size
for epoch in range(epochs):
    perms = jax.random.permutation(rng, train['image'].shape[0])
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    batch_metrics = list()
    for perm in tqdm(perms):
        image_batch = train['image'][perm, ...]
        label_batch = train['label'][perm, ...]
        state, rng, metrics = coder.train_step(state, rng, image_batch, label_batch)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                        for metric in batch_metrics_np[0]}
    write_summary(summary_writer, epoch_metrics_np, rng, epoch, True, False)
    if (epoch + 1) % test_interval == 0 or epoch == 0:
        rng, eval_rng = random.split(rng)
        eval_metrics = coder.evaluate(state, eval_rng, test['image'], test['label'])
        eval_metrics_np = jax.device_get(eval_metrics)
        write_summary(summary_writer, eval_metrics_np, eval_rng, epoch, False, True)
    









