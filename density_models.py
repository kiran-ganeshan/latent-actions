from re import X
import jax
import numpy as np
import os, sys
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, lax, random, value_and_grad, partial
from jax.experimental.host_callback import id_print
from jax.ops import index, index_update
from jax.nn import one_hot
from jax.nn.initializers import variance_scaling
from tqdm import tqdm
from absl import flags
from flax.metrics.tensorboard import SummaryWriter
from flax.linen.initializers import zeros
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass
from dataclasses import dataclass as py_dataclass
from enum import Enum
import datetime
from typing import Sequence
from utils import get_datasets, write_summary, write_data
from optax import adam
import wandb


FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('test_interval', 2, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('num_values', 10, 'Number of values for categorical latent variables.')
flags.DEFINE_integer('embedding_dim', 3, 'Dimension of embedding space.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('prng_key', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('temp_interval', 2, 'Temperature update interval (epochs).')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('max_temp', 0.5, 'Maximum temperature.')
flags.DEFINE_float('temp_rate', 0.1, 'Rate of temperature decrease (per epoch).')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
flags.DEFINE_float('commitment_cost', 0.25, 'VQVAE Quantizer commitment cost.')
flags.DEFINE_float('eps', 1e-20, 'Stability epsilon in jnp.log.')


def kl_loss(mu, log_sigma):
    kl_exp = 1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma)
    return -jnp.mean(jnp.sum(kl_exp, axis=-1))

def bce_loss(inputs, outputs):
    bce_exp = inputs * outputs + (1 - inputs) * jnp.log(-jnp.expm1(outputs) + 1e-9)
    return -jnp.mean(jnp.sum(bce_exp, axis=(1, 2, 3)))

def concat_labels(inputs, labels):
    inputs = inputs.reshape((inputs.shape[0], -1))
    inputs = jnp.concatenate([inputs, one_hot(labels, 10)], axis=1)
    return inputs


class TrainState(TrainState):
    batch_stats : FrozenDict[str, any]


@dataclass
class MADETrainState:
    made_state : TrainState
    epoch : int
    
    @classmethod
    def create(cls, module, params, optim):
        made_state = TrainState.create(apply_fn=module.apply,
                                       params=params,
                                       tx=optim,
                                       batch_stats={})
        return MADETrainState(made_state=made_state, epoch=0)
        
    def next_epoch(self):
        return MADETrainState(made_state=self.made_state, epoch=self.epoch + 1)
    
    def apply_gradients(self, grads):
        new_state = self.made_state.apply_gradients(grads=grads)
        return MADETrainState(made_state=new_state, epoch=self.epoch)
        


@dataclass
class VAETrainState:
    enc_state : TrainState
    dec_state : TrainState
    epoch : int
    
    @classmethod
    def create(cls, enc_module, enc_params, enc_optim, dec_module, dec_params, dec_optim):
        enc_state = TrainState.create(apply_fn=enc_module.apply, 
                                      params=enc_params,
                                      tx=enc_optim,
                                      batch_stats={})
        dec_state = TrainState.create(apply_fn=dec_module.apply,
                                      params=dec_params,
                                      tx=dec_optim,
                                      batch_stats={})
        return VAETrainState(enc_state=enc_state, dec_state=dec_state, epoch=0)
    
    def next_epoch(self):
        return VAETrainState(enc_state=self.enc_state, 
                             dec_state=self.dec_state, 
                             epoch=self.epoch + 1)
    
    def apply_gradients(self, enc_grads, dec_grads):
        new_enc_state = self.enc_state.apply_gradients(grads=enc_grads)
        new_dec_state = self.dec_state.apply_gradients(grads=dec_grads)
        return VAETrainState(enc_state=new_enc_state, dec_state=new_dec_state, epoch=self.epoch)
    
    
@dataclass
class GSVAETrainState(VAETrainState):
    temp : float
    temp_interval : int
    max_temp : float
    temp_rate : float
    
    def create(cls, enc_module, enc_params, enc_optim,
               dec_module, dec_params, dec_optim,
               temp_interval, max_temp, temp_rate):
        vae_train_state = VAETrainState.create(enc_module, enc_params, enc_optim,
                                               dec_module, dec_params, dec_optim)
        return GSVAETrainState(enc_state=vae_train_state.enc_state,
                               dec_state=vae_train_state.dec_state,
                               temp=max_temp, temp_interval=temp_interval,
                               temp_rate=temp_rate, max_temp=max_temp,
                               epoch=0)
    
    def next_epoch(self):
        adjust_temp = ((self.epoch + 1) % self.temp_interval == 0)
        new_temp = lambda epoch: self.max_temp * jnp.exp(-self.temp_rate * epoch)
        new_temp_val = lax.cond(adjust_temp, new_temp, lambda e: self.temp, self.epoch)
        return GSVAETrainState(enc_state=self.enc_state, 
                               dec_state=self.dec_state, 
                               temp=new_temp_val,
                               max_temp=self.max_temp,
                               temp_interval=self.temp_interval,
                               temp_rate=self.temp_rate,
                               epoch=self.epoch + 1)
        
    def apply_gradients(self, enc_grads, dec_grads):
        new_enc_state = self.enc_state.apply_gradients(grads=enc_grads)
        new_dec_state = self.dec_state.apply_gradients(grads=dec_grads)
        return GSVAETrainState(enc_state=new_enc_state, dec_state=new_dec_state, 
                               temp=self.temp, temp_interval=self.temp_interval, 
                               max_temp=self.max_temp, temp_rate=self.temp_rate, 
                               epoch=self.epoch)

    
@dataclass
class VQVAETrainState(VAETrainState):
    vq_state : TrainState
    encodings : jnp.array
    latent_dim : int
    num_values : int
    
    @classmethod
    def create(cls, enc_module, enc_params, enc_optim,
               dec_module, dec_params, dec_optim, 
               vq_module, vq_params, vq_optim, 
               latent_dim, num_values):
        vae_train_state = VAETrainState.create(enc_module, enc_params, enc_optim,
                                               dec_module, dec_params, dec_optim)
        vq_state = TrainState.create(apply_fn=vq_module.apply,
                                     params=vq_params,
                                     tx=vq_optim,
                                     batch_stats={})
        return VQVAETrainState(enc_state=vae_train_state.enc_state,
                               dec_state=vae_train_state.dec_state,
                               vq_state=vq_state, 
                               encodings=jnp.array([]).reshape((0, latent_dim, num_values)), 
                               latent_dim=latent_dim, 
                               num_values=num_values, 
                               epoch=0)

    def next_epoch(self):
        no_encodings = jnp.array([]).reshape((0, self.latent_dim, self.num_values))
        return VQVAETrainState(enc_state=self.enc_state, 
                               dec_state=self.dec_state, 
                               vq_state=self.vq_state,
                               encodings=no_encodings,
                               latent_dim=self.latent_dim,
                               num_values=self.num_values,
                               epoch=self.epoch + 1)
    
    def apply_gradients(self, enc_grads, dec_grads, vq_grads, encodings):
        enc_state = self.enc_state.apply_gradients(grads=enc_grads)
        dec_state = self.dec_state.apply_gradients(grads=dec_grads)
        vq_state = self.vq_state.apply_gradients(grads=vq_grads)
        encodings = jnp.concatenate([self.encodings, encodings], 0)
        return VQVAETrainState(enc_state=enc_state, 
                               dec_state=dec_state, 
                               vq_state=vq_state, 
                               encodings=encodings,
                               latent_dim=self.latent_dim,
                               num_values=self.num_values,
                               epoch=self.epoch)
    
    
class MaskType(Enum):
    input = 1
    hidden = 2
    output = 3


@jax.util.cache()
def get_mask(in_dim, out_dim, rand_dim, mask_type : MaskType):
    if mask_type == MaskType.input:
        in_degrees = jnp.arange(in_dim) % rand_dim
    else:
        in_degrees = jnp.arange(in_dim) % (rand_dim - 1)
    if mask_type == MaskType.output:
        out_degrees = jnp.arange(out_dim) % rand_dim - 1
    else:
        out_degrees = jnp.arange(out_dim) % (rand_dim - 1)
    in_degrees = jnp.expand_dims(in_degrees, 0)
    out_degrees = jnp.expand_dims(out_degrees, -1)
    return (out_degrees >= in_degrees).astype(jnp.float32).transpose()


class MaskedDense(nn.Dense):
    latent_dim : int = 10
    mask_type : MaskType = MaskType.hidden
    use_bias : bool = False
    
    @nn.compact
    def __call__(self, x):
        x = jnp.asarray(x, self.dtype)
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = kernel * get_mask(*kernel.shape, self.latent_dim, self.mask_type)
        y = lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())), self.precision)
        bias = self.param('bias', self.bias_init, (1, self.features))
        add_bias, no_bias = lambda y: y + bias, lambda y: y
        y = lax.cond(self.use_bias, add_bias, no_bias, y)
        return y 


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
    
    
class MaskedMLP(nn.Module):
    
    model_name : str = 'masked_mlp'
    hidden_sizes : Sequence[int] = (784,)
    latent_dim : int = 784
    num_values : int = 2
    
    @nn.compact
    def __call__(self, x):
        x = x.swapaxes(1, 2).reshape((-1, self.latent_dim * self.num_values))
        x = nn.relu(MaskedDense(features=self.hidden_sizes[0],
                                latent_dim=self.latent_dim,
                                mask_type=MaskType.input,
                                use_bias=False)(x))
        for size in self.hidden_sizes[1:]:
            x = nn.relu(MaskedDense(features=size,
                                    latent_dim=self.latent_dim,
                                    mask_type=MaskType.hidden,
                                    use_bias=False)(x))
        x = MaskedDense(features=self.latent_dim * self.num_values,
                        latent_dim=self.latent_dim,
                        mask_type=MaskType.output,
                        use_bias=True)(x)
        x = x.reshape((-1, self.num_values, self.latent_dim)).swapaxes(1, 2)
        x = nn.log_softmax(x + FLAGS.eps, axis=-1)
        return x


class Decoder(MLP):
    hidden_sizes : Sequence[int] = (100, 256)
    input_size : int = 10
    output_size : int = 784

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        x = nn.log_sigmoid(x).reshape((-1, 28, 28, 1))
        return x


class Encoder(MLP):
    latent_dim : int = 10
    hidden_sizes : Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        mu = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        log_sigma = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        return mu, log_sigma


class Classifier(MLP):
    hidden_sizes : Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        return nn.softmax(x)


class GSEncoder(MLP):
    latent_dim : int = 10
    latent_vars : int = 20
    hidden_sizes : Sequence[int] = (256,)
    input_size : int = 784
    output_size : int = 100

    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        x = nn.Dense(self.latent_dim * self.latent_vars, kernel_init=zeros)(x)
        x = x.reshape((x.shape[0], self.latent_dim, self.latent_vars))
        return x
    
    
class Quantizer(nn.Module):
    embedding_dim : int = 10
    num_embeddings : int = 10
    commitment_cost : float = 0.25

    @nn.compact
    def __call__(self, x):
        e = self.param('embeddings', variance_scaling(1.0, 'fan_in', 'uniform'),
                       (self.embedding_dim, self.num_embeddings))
        input_shape = x.shape[:-1]
        x = x.reshape((-1, self.embedding_dim))
        dist = jnp.sum(x * x, -1, keepdims=True) + jnp.sum(e * e, 0, keepdims=True) - 2 * x @ e
        enc_idx = jnp.argmax(-dist, 1)
        enc = one_hot(enc_idx, self.num_embeddings)
        quantized = jax.device_put(e.T)[(enc_idx.reshape(input_shape),)]
        x = x.reshape((*input_shape, self.embedding_dim))
        e_latent_loss = jnp.mean((lax.stop_gradient(quantized) - x)**2)
        q_latent_loss = jnp.mean((quantized - lax.stop_gradient(x))**2)
        loss = q_latent_loss #+ self.commitment_cost * e_latent_loss
        quantized = x + lax.stop_gradient(quantized - x)
        avg_probs = jnp.mean(enc, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        aux = {
            'loss': loss,
            'encoding': enc,
            'encoding_index': enc_idx,
            'avg_probs': avg_probs,
            'perplexity': perplexity
        }
        return quantized, aux


@dataclass
class VAELearner:
    model_name : str = 'vae'
    latent_dim : int = 10
    learning_rate : float = 1e-4
    beta1 : float = 0.5
    beta2 : float = 0.9
    beta : float = 0.5
    num_classes : int = 10
    image_size : int = 784
    image_shape : Sequence[int] = (28, 28, 1)
    
    @partial(jit, static_argnums=0)
    def initial_state(self, rng):
        enc_rng, dec_rng = random.split(rng)
        encoder = Encoder(latent_dim=self.latent_dim)
        decoder = Decoder(input_size=self.latent_dim)
        enc_params = encoder.init(enc_rng, jnp.ones((10,) + self.image_shape, jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim), jnp.float32))
        make_optim = lambda: adam(self.learning_rate, self.beta1, self.beta2)
        return VAETrainState.create(encoder, enc_params, make_optim(), decoder, dec_params, make_optim())
        
    def compute_loss(self, train_state, enc_params, dec_params, rng, inputs, labels, train=True):
        mu, log_sigma = train_state.enc_state.apply_fn(enc_params, inputs)
        r = random.normal(rng, (mu.shape[0], self.latent_dim))
        codes = mu + r * jnp.exp(log_sigma)
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), kl_loss(mu, log_sigma), aux

    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, inputs, labels, train=True):
        def loss_fn(enc_params, dec_params, rng):
            reconst_loss, kl_penalty, aux = self.compute_loss(train_state, enc_params, dec_params, 
                                                              rng, inputs, labels, train)
            metrics = {'loss': reconst_loss + kl_penalty, 
                       'reconst_loss': reconst_loss, 
                       'penalty_kl_loss': kl_penalty}
            return reconst_loss + self.beta * kl_penalty, metrics
        
        step_rng, rng = random.split(rng)
        enc_params, dec_params = train_state.enc_state.params, train_state.dec_state.params
        val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))
        (loss, batch_stats), (enc_grads, dec_grads) = val_and_grad(enc_params, dec_params, step_rng)
        new_train_state = train_state.apply_gradients(enc_grads, dec_grads)
        return new_train_state, rng, batch_stats

    @partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        codes = random.normal(rng, (25, self.latent_dim))
        dec_params = train_state.dec_state.params
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        return reconst, jnp.zeros((25,))
                        #dummy labels so evaluate is compatible with classVAE
    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, inputs, labels):
        loss_rng, generate_rng = random.split(rng)
        reconst_loss, kl_penalty, aux = self.compute_loss(train_state,
                                                          train_state.enc_state.params, 
                                                          train_state.dec_state.params,
                                                          loss_rng, inputs, labels)
        generated, gen_labels = self.generate(train_state, generate_rng)
        metrics = {'loss': reconst_loss + self.beta * kl_penalty, 
                   'reconst_loss': reconst_loss, 
                   'penalty_kl_loss': kl_penalty}
        data = {'generated': jnp.exp(generated),
                'generated_label': gen_labels, **aux}
        return metrics, data


@dataclass
class ClassVAELearner(VAELearner):
    model_name : str = 'class_vae'

    @partial(jit, static_argnums=0)
    def initial_state(self, rng):
        enc_rng, dec_rng = random.split(rng)
        encoder = Encoder(latent_dim=self.latent_dim, input_size=self.image_size + self.num_classes)
        decoder = Decoder(input_size=self.latent_dim + self.num_classes)
        enc_params = encoder.init(enc_rng, jnp.ones((10, self.image_size), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim + self.num_classes), jnp.float32))
        make_optim = lambda: adam(self.learning_rate, self.beta1, self.beta2)
        return VAETrainState.create(encoder, enc_params, make_optim(), decoder, dec_params, make_optim())
        
    def compute_loss(self, train_state, enc_params, dec_params, rng, inputs, labels, train=True):
        inputs = concat_labels(inputs, labels)
        mu, log_sigma = train_state.enc_state.apply_fn(enc_params, inputs)
        r = random.normal(rng, (mu.shape[0], self.latent_dim))
        codes = mu + r * jnp.exp(log_sigma)
        codes = concat_labels(codes, labels)
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        inputs = inputs[:, :self.image_size].reshape((-1,) + self.image_shape)
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), kl_loss(mu, log_sigma, self.beta), aux

    @partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        codes = random.normal(rng, (25, self.latent_dim))
        labels = random.randint(rng, (25,), 0, 10)
        codes = jnp.concatenate([codes, one_hot(labels, self.num_classes)], axis=1)
        dec_params = train_state.dec_state.params
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        return reconst, labels


@dataclass
class GSVAELearner(VAELearner):
    model_name : str = 'gsvae'
    max_temp : float = 0.5
    temp_rate : float = 0.1
    temp_interval : int = 2
    latent_vars : int = 20

    @partial(jit, static_argnums=0)
    def initial_state(self, rng):
        enc_rng, dec_rng = random.split(rng)
        encoder = GSEncoder(latent_dim=self.latent_dim, latent_vars=self.latent_vars)
        decoder = Decoder(input_size=self.latent_dim * self.latent_vars)
        enc_params = encoder.init(enc_rng, jnp.ones((10, 784), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim * self.latent_vars), jnp.float32))
        make_optim = lambda: adam(self.learning_rate, self.beta1, self.beta2)
        return GSVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(), 
                                      self.temp_interval, self.max_temp, self.temp_rate)
        
    def compute_loss(self, train_state, enc_params, dec_params, rng, inputs, labels, train=True):
        logprobs = train_state.enc_state.apply_fn(enc_params, inputs)
        g = -jnp.log(-jnp.log(random.uniform(rng, logprobs.shape) + FLAGS.eps) + FLAGS.eps)
        train_sample = lambda x: nn.softmax((x + g)/train_state.temp, axis=1)
        test_sample = lambda x: one_hot(jnp.argmax(x, axis=1), self.latent_dim).swapaxes(1, 2)
        codes = lax.cond(train, train_sample, test_sample, logprobs)
        codes = codes.reshape((codes.shape[0], -1))
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), 0, aux

    # Not sure how to generate, will start by randomly sampling from latents but this def won't work
    @partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        codes = random.randint(rng, (25, self.latent_vars), 0, self.latent_dim)
        codes = one_hot(codes, self.latent_dim).reshape((codes.shape[0], -1))
        labels = random.randint(rng, (25,), 0, 10)
        dec_params = train_state.dec_state.params
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        return reconst, labels


@dataclass
class ClassGSVAELearner(GSVAELearner):
    model_name : str = 'class_gsvae'

    @partial(jit, static_argnums=0)
    def initial_state(self, rng):
        enc_rng, dec_rng = random.split(rng)
        decoder_input_size = self.latent_dim * self.latent_vars + self.num_classes
        encoder = GSEncoder(latent_dim=self.latent_dim, latent_vars=self.latent_vars,
                            input_size=self.image_size + self.num_classes)
        decoder = Decoder(input_size=decoder_input_size)
        enc_params = encoder.init(enc_rng, jnp.ones((10, self.image_size + self.num_classes), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, decoder_input_size), jnp.float32))
        make_optim = lambda: adam(self.learning_rate, self.beta1, self.beta2)
        return GSVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(), 
                                      self.temp_interval, self.max_temp, self.temp_rate)

    def compute_loss(self, train_state, enc_params, dec_params, rng, inputs, labels, train=True):
        inputs = concat_labels(inputs, labels)
        logprobs = train_state.enc_state.apply_fn(enc_params, inputs)
        g = -jnp.log(-jnp.log(random.uniform(rng, logprobs.shape) + FLAGS.eps) + FLAGS.eps)
        train_sample = lambda x: nn.softmax((x + g)/train_state.temp, axis=1)
        test_sample = lambda x: one_hot(jnp.argmax(x, axis=1), self.latent_dim).swapaxes(1, 2)
        codes = lax.cond(train, train_sample, test_sample, logprobs)
        codes = concat_labels(codes, labels)
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        inputs = inputs[:, :self.image_size].reshape((-1,) + self.image_shape)
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), 0, aux

    @partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        codes = random.randint(rng, (25, self.latent_vars), 0, self.latent_dim)
        codes = one_hot(codes, self.latent_dim)
        labels = random.randint(rng, (25,), 0, 10)
        codes = concat_labels(codes, labels)
        dec_params = train_state.dec_state.params
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        return reconst, labels
    
    
@dataclass
class MADELearner:  
    
    model_name : str = 'made'
    latent_dim : int = 784
    num_values : int = 2
    learning_rate : float = 1e-4
    beta1 : float = 0.9
    beta2 : float = 0.5
    
    #@partial(jit, static_argnums=0)
    def initial_state(self, rng):
        made = MaskedMLP(latent_dim=self.latent_dim, num_values=self.num_values)
        params = made.init(rng, jnp.ones((10, self.latent_dim, self.num_values), jnp.float32))
        optim = adam(self.learning_rate, self.beta1, self.beta2)
        return MADETrainState.create(made, params, optim)
    
    def compute_loss(self, train_state, params, rng, inputs, train=True):
        logprobs = train_state.made_state.apply_fn(params, inputs)
        return jnp.mean(jnp.sum(-inputs * logprobs, axis=(1, 2)))
    
    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, inputs, train=True):
        inputs = one_hot(inputs.reshape((-1, self.latent_dim)), self.num_values)
        def loss_fn(params, rng):
            loss = self.compute_loss(train_state, params, rng, inputs, train)
            metrics = {'loss': loss}
            return loss, metrics
        step_rng, rng = random.split(rng)
        params = train_state.made_state.params
        (loss, batch_stats), grads = value_and_grad(loss_fn, has_aux=True)(params, step_rng)
        new_train_state = train_state.apply_gradients(grads=grads)
        return new_train_state, rng, batch_stats
    
    #@partial(jit, static_argnums=0)
    def generate(self, train_state, rng):
        rng = random.split(rng, self.latent_dim)
        samples = jnp.ones((25, self.latent_dim, self.num_values))
        def sample_next_dim(i, samples):
            logprobs = train_state.made_state.apply_fn(train_state.made_state.params, samples)
            next_dim = random.categorical(rng[i, :], logprobs[:, i, :], axis=-1)
            return index_update(samples, index[:, i, :], one_hot(next_dim, self.num_values))
        samples = lax.fori_loop(0, self.latent_dim, jit(sample_next_dim), samples)
        return samples
    
    @partial(jit, static_argnums=0)
    def greedy_generate(self, train_state, rng, inputs):
        logprobs = train_state.made_state.apply_fn(train_state.made_state.params, inputs)
        sample = random.categorical(rng, logprobs, axis=-1)
        return one_hot(sample, self.num_values)
    
    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, inputs, labels):
        loss_rng, generate_rng = random.split(rng)
        inputs = one_hot(inputs.reshape((-1, self.latent_dim)), self.num_values)
        loss = self.compute_loss(train_state, train_state.made_state.params,
                                 loss_rng, inputs, train=False)
        generated = self.generate(train_state, generate_rng)
        generated = jnp.argmax(generated, axis=-1).reshape((-1, 28, 28, 1))
        inputs = jnp.argmax(inputs, axis=-1).reshape((-1, 28, 28, 1))
        metrics = {'loss': loss}
        data = {'generated': generated, 
                'image': inputs, 
                'output': inputs, 
                'label': jnp.zeros((inputs.shape[0],)), 
                'generated_label': jnp.zeros((generated.shape[0],))}
        return metrics, data


@dataclass
class VQVAELearner(VAELearner):
    model_name : str = 'vqvae'
    embedding_dim : int = 10
    num_values : int = 10
    commitment_cost : float = 0.25
    beta : float = 0.5
    made_learning_rate : float = 1e-5
    made_beta1 : float = 0.9
    made_beta2 : float = 0.5
    made_epochs : int = 10
    made_batch : int = 60

    def initial_state(self, rng):
        enc_rng, dec_rng, vq_rng = random.split(rng, 3)
        encoder = MLP(input_size=self.image_size, 
                      output_size=self.latent_dim * self.embedding_dim)
        decoder = Decoder(input_size=self.latent_dim * self.embedding_dim)
        quantizer = Quantizer(embedding_dim=self.embedding_dim,
                              num_embeddings=self.num_values,
                              commitment_cost=self.commitment_cost)
        enc_params = encoder.init(enc_rng, jnp.ones((10,) + self.image_shape, jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim, self.embedding_dim), jnp.float32))
        vq_params = quantizer.init(vq_rng, jnp.ones((10, self.latent_dim, self.embedding_dim), jnp.float32))
        make_optim = lambda: adam(self.learning_rate, self.beta1, self.beta2)
        print(self.latent_dim)
        return VQVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(),
                                      quantizer, vq_params, make_optim(),
                                      self.latent_dim, self.num_values)

    def compute_loss(self, train_state, enc_params, dec_params, vq_params, 
                     rng, inputs, labels, train=True):
        latents = train_state.enc_state.apply_fn(enc_params, inputs)
        latents = latents.reshape((-1, self.latent_dim, self.embedding_dim))
        codes, vq_aux = train_state.vq_state.apply_fn(vq_params, latents)
        #codes = latents
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        aux = {'image': inputs, 'label': labels, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), vq_aux.pop('loss'), {**aux, **vq_aux}
        #return bce_loss(inputs, reconst), 0, aux
    
    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, inputs, labels, train=True):
        def loss_fn(enc_params, dec_params, vq_params, rng):
            reconst_loss, kl_penalty, aux = self.compute_loss(train_state, enc_params, dec_params, 
                                                              vq_params, rng, inputs, labels, train)
            metrics = {'loss': reconst_loss + self.beta * kl_penalty, 
                       'reconst_loss': reconst_loss, 
                       'penalty_kl_loss': kl_penalty,
                       'encoding': aux['encoding']}
            return reconst_loss + self.beta * kl_penalty, metrics
        
        step_rng, rng = random.split(rng)
        val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))
        (loss, stats), (e_grads, d_grads, q_grads) = val_and_grad(train_state.enc_state.params, 
                                                                  train_state.dec_state.params, 
                                                                  train_state.vq_state.params, 
                                                                  step_rng)
        enc = stats['encoding'].reshape((-1, self.latent_dim, self.num_values))
        metric_names = ['loss', 'reconst_loss', 'penalty_kl_loss']
        stats = {name: stats[name] for name in metric_names}
        new_train_state = train_state.apply_gradients(enc_grads=e_grads, dec_grads=d_grads, 
                                                      vq_grads=q_grads, encodings=enc)
        return new_train_state, rng, stats

    def make_made(self):
        hidden_sizes = 2 * (self.latent_dim * self.num_values,)
        made = MADELearner(latent_dim=self.latent_dim,
                           num_values=self.num_values,
                           learning_rate=self.made_learning_rate,
                           beta1=self.made_beta1,
                           beta2=self.made_beta2,
                           hidden_sizes=hidden_sizes)
        return made

    @partial(jit, static_argnums=0)
    def generate(self, train_state, made_train_state, rng):
        samples = self.make_made().generate(made_train_state, rng)
        embeddings = train_state.vq_state.params['params']['embeddings']
        idx = jnp.argmax(samples.reshape((-1, self.latent_dim, self.embedding_dim)), -1)
        quantized = jax.device_put(embeddings.T)[(idx,)]
        outputs = train_state.dec_state.apply_fn(train_state.dec_state.params, quantized)
        return jnp.exp(outputs), jnp.zeros((25,))
    
    def train_made(self, encodings, rng, epochs):
        made = self.make_made()
        rng, init_rng = random.split(rng)
        state = made.initial_state(init_rng)
        
        path = os.path.join(FLAGS.save_dir, made.model_name)
        path = os.path.join(path, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        summary_writer = SummaryWriter(path)
        
        steps_per_epoch = encodings.shape[0] // self.made_batch
        for epoch in range(epochs):
            perms = jax.random.permutation(rng, encodings.shape[0])
            perms = perms[:steps_per_epoch * self.made_batch]
            perms = perms.reshape((steps_per_epoch, self.made_batch))
            batch_metrics = list()
            for perm in tqdm(perms, leave=False):
                batch = encodings[perm, ...]
                state, rng, metrics = made.train_step(state, rng, batch)
                batch_metrics.append(metrics)
            batch_metrics_np = jax.device_get(batch_metrics)
            epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                                for metric in batch_metrics_np[0]}
            write_summary(summary_writer, epoch_metrics_np, epoch, True, False)
            state = state.next_epoch()
        return state
            
    def evaluate(self, train_state, rng, inputs, labels):
        loss_rng, made_rng, generate_rng = random.split(rng, 3)
        reconst_loss, kl_penalty, aux = self.compute_loss(train_state,
                                                          train_state.enc_state.params, 
                                                          train_state.dec_state.params,
                                                          train_state.vq_state.params,
                                                          loss_rng, inputs, labels)
        made_state = self.train_made(train_state.encodings, made_rng, self.made_epochs)
        generated, gen_labels = self.generate(train_state, made_state, generate_rng)
        metrics = {'loss': reconst_loss + kl_penalty, 
                   'reconst_loss': reconst_loss, 
                   'penalty_kl_loss': kl_penalty,
                   'perplexity': aux.pop('perplexity')}
        data = {'generated': jnp.exp(generated),
                'generated_label': gen_labels, **aux}
        return metrics, data


def main():
    #wandb.init()
    FLAGS(sys.argv) 

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    test_interval = FLAGS.test_interval
    train, test = get_datasets()
    coder = VQVAELearner(learning_rate=FLAGS.learning_rate,
                         beta1=FLAGS.beta2,
                         beta2=FLAGS.beta1,
                         beta=FLAGS.beta,
                         commitment_cost=FLAGS.commitment_cost,
                         latent_dim=FLAGS.latent_dim,
                         embedding_dim=FLAGS.embedding_dim,
                         num_values=FLAGS.num_values)
    rng, init_rng = random.split(random.PRNGKey(FLAGS.prng_key))
    state = coder.initial_state(init_rng)

    path = os.path.join(FLAGS.save_dir, coder.model_name)
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
        write_summary(summary_writer, epoch_metrics_np, epoch, True)
        if epoch % test_interval == 0 or epoch == epochs - 1:
            rng, eval_rng = random.split(rng)
            eval_metrics, eval_data = coder.evaluate(state, eval_rng, test['image'], test['label'])
            eval_metrics_np, eval_data_np = jax.device_get(eval_metrics), jax.device_get(eval_data)
            write_summary(summary_writer, eval_metrics_np, epoch, False)
            write_data(summary_writer, eval_data_np, epoch, False)
        state = state.next_epoch()
        
    #wandb.finish()

if __name__ == "__main__":
    main()





