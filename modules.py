from re import X
import jax
from jax import numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Sequence, Callable
from jax.interpreters.xla import DeviceArray
from enum import Enum
from jax.nn.initializers import variance_scaling
from jax.nn import one_hot
from flax.linen.initializers import zeros

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


class Module(nn.Module):
    
    @property 
    def input_shape(self):
        pass


class MLP(Module):

    model_name : str = 'mlp'
    hidden_sizes : Sequence[int] = (100,)
    input_size : int = 784
    output_size : int = 10
    activation : Callable[[DeviceArray], DeviceArray] = lambda x: x
    
    @property
    def input_shape(self):
        return (self.input_size,)

    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, self.input_size))
        return self.mlp(x)

    def mlp(self, x):
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.Dense(features=self.output_size)(x)
        x = self.activation(x)
        return x
    
class Encoder(MLP):
    
    latent_dim : int = 10
    
    @nn.compact
    def __call__(self, x):
        x = self.mlp(x.reshape((-1, self.input_size)))
        mu = nn.Dense(features=self.latent_dim, kernel_init=zeros)(x)
        log_sigma = nn.Dense(features=self.latent_dim, kernel_init=zeros)(x)
        return mu, log_sigma
    
    
class MaskedMLP(Module):
    
    model_name : str = 'masked_mlp'
    hidden_sizes : Sequence[int] = (784,)
    latent_dim : int = 784
    num_values : int = 2
    
    @property 
    def input_shape(self):
        return (self.latent_dim, self.num_values)
    
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
        x = nn.log_softmax(x + 1e-20, axis=-1)
        return x
    
    
class CondMaskedMLP(MaskedMLP):
    
    model_name : str = 'masked_mlp'
    cond_size : int = 10
    
    @nn.compact
    def __call__(self, x, y):
        x = x.swapaxes(1, 2).reshape((-1, self.latent_dim * self.num_values))
        y = y.reshape((-1, self.cond_size))
        x = nn.relu(MaskedDense(features=self.hidden_sizes[0],
                                latent_dim=self.latent_dim,
                                mask_type=MaskType.input,
                                use_bias=False)(x))
        y = nn.relu(nn.Dense(features=self.hidden_sizes[0])(y))
        x = x + y
        for size in self.hidden_sizes[1:]:
            x = nn.relu(MaskedDense(features=size,
                                    latent_dim=self.latent_dim,
                                    mask_type=MaskType.hidden,
                                    use_bias=False)(x))
            y = nn.relu(nn.Dense(features=size)(y))
            x = x + y
        x = MaskedDense(features=self.latent_dim * self.num_values,
                        latent_dim=self.latent_dim,
                        mask_type=MaskType.output,
                        use_bias=True)(x)
        y = nn.Dense(features=self.latent_dim * self.num_values)(y)
        x = x + y
        x = x.reshape((-1, self.num_values, self.latent_dim)).swapaxes(1, 2)
        x = nn.log_softmax(x + 1e-20, axis=-1)
        return x
    
    
class DoubleQDecoder(Module):
    
    input_size : int
    model_name : str = 'double_clipped_q'
    hidden_sizes : Sequence[int] = (784,)
    activation : Callable[[DeviceArray], DeviceArray] = lambda x: x
    
    @property
    def input_shape(self):
        return (self.input_size,)
    
    @nn.compact
    def __call__(self, codes):
        x, y = codes, codes
        x = MLP(hidden_sizes=self.hidden_sizes,
                input_size=self.input_size,
                output_size=1,
                activation=self.activation)(x)
        y = MLP(hidden_sizes=self.hidden_sizes,
                input_size=self.input_size,
                output_size=1,
                activation=self.activation)(y)
        return jnp.concatenate([x, y], axis=-1)
    
    
class Quantizer(Module):
    embedding_dim : int = 10
    num_embeddings : int = 10
    commitment_cost : float = 0.25
    
    @property
    def input_shape(self):
        return (self.num_embeddings, self.embedding_dim)

    @nn.compact
    def __call__(self, x, train=True):
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
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + lax.stop_gradient(quantized - x)
        avg_probs = jnp.mean(enc.reshape((*input_shape, -1)), 0)
        over_latents = jnp.mean(avg_probs, tuple(range(len(input_shape) - 1)))
        perplexity = jnp.exp(-jnp.sum(over_latents * jnp.log(over_latents + 1e-20)))
        aux = {
            'encoding': enc,
            'encoding_index': enc_idx,
            'avg_probs': avg_probs,
            'perplexity': perplexity,
            'centers': e.T
        }
        return quantized, loss, aux
    
    
class KMeansQuantizer(Module):
    embedding_dim : int = 10
    num_embeddings : int = 10
    commitment_cost : float = 0.25
    momentum : float = 0.9
    
    @property
    def input_shape(self):
        return (self.num_embeddings, self.embedding_dim)

    @nn.compact
    def __call__(self, x, train=True):
        input_shape = x.shape[:-1]
        x = x.reshape((-1, self.embedding_dim))
        embeddings = self.variable('embedding_vars', 'embeddings', 
                                   lambda s: variance_scaling(1.0, 'fan_in', 'uniform')(
                                       self.make_rng('params'), s),
                                   (self.embedding_dim, self.num_embeddings))
        cluster_size = self.variable('embedding_vars', 'cluster_size', 
                                     lambda s: jnp.zeros(s), (self.num_embeddings,))
        unnormalized = self.variable('embedding_vars', 'unnormalized_embeds', 
                                     lambda s: jnp.zeros(s), (self.embedding_dim, self.num_embeddings))
        e, cs, un = embeddings.value, cluster_size.value, unnormalized.value
        dist = jnp.sum(x * x, 1, keepdims=True) + jnp.sum(e * e, 0, keepdims=True) - 2 * x @ e
        enc_idx = jnp.argmax(-dist, 1)
        enc = one_hot(enc_idx, self.num_embeddings)
        quantized = jax.device_put(e.T)[(enc_idx.reshape(input_shape),)]
        def update(_):
            new_cs = (1 - self.momentum) * jnp.sum(enc, axis=0) + self.momentum * cs
            new_un = (1 - self.momentum) * x.T @ enc + self.momentum * un
            n = jnp.sum(new_cs)
            stable_cs = (new_cs + 1e-20) / (n + self.num_embeddings * 1e-20) * n
            new_e = new_un / stable_cs.reshape((1, -1))
            return new_e, new_cs, new_un
        initializing = self.is_mutable_collection('params')
        no_update = lambda _: (e, cs, un)
        new_e, new_cs, new_un = lax.cond(train, update, no_update, None)
        full_update = lambda _: (new_e, new_cs, new_un)
        new_e, new_cs, new_un = lax.cond(initializing, no_update, full_update, None)
        embeddings.value, cluster_size.value, unnormalized.value = new_e, new_cs, new_un
        x = x.reshape((*input_shape, self.embedding_dim))
        loss = self.commitment_cost * jnp.mean((lax.stop_gradient(quantized) - x)**2)
        quantized = x + lax.stop_gradient(quantized - x)
        avg_probs = jnp.mean(enc.reshape((*input_shape, -1)), 0)
        over_latents = jnp.mean(avg_probs, tuple(range(len(input_shape) - 1)))
        perplexity = jnp.exp(-jnp.sum(over_latents * jnp.log(over_latents + 1e-20)))
        aux = {
            'encoding': enc,
            'encoding_index': enc_idx,
            'avg_probs': avg_probs,
            'perplexity': perplexity,
            'centers': embeddings.value.T
        }
        return quantized, loss, aux