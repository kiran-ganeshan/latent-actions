import numpy as np
from jax import numpy as jnp
from flax import linen as nn
from jax import jit, lax, random, value_and_grad, partial
from jax.ops import index, index_update
from jax import device_get, device_put
from jax.experimental.host_callback import id_print
from jax.experimental.optimizers import exponential_decay as decay
from jax.nn import one_hot
from tqdm import tqdm
from flax.struct import dataclass
from typing import Sequence
from utils import kl_loss, bce_loss
from modules import *
from states import *
from optax import adam

class Submodules:
    
    submodules = []
    
    def __init__(self, module):
        Submodules.submodules.append(module)
        self.module = module
        
    def __call__(self, *args, **kwargs):
        return self.module(args, kwargs)
    
submodules = []
def submodule_decorator():
    def submodule(f):
        super(Learner).submodules[f.__name__] = f()
        return property(f)
    return submodule

def empty_submodules():
    submodules = []


@dataclass
class Learner:
    submodules : 'dict[str, Module]' = {}
    submodule = submodule_decorator(submodules)
    
    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, *data):
        def loss_fn(params, rng):
            losses, metrics, aux, new_vars = self.compute_loss(train_state, params, rng, *data, train=True)
            loss = sum(losses.values())
            return loss, {'metrics': {'loss': loss, **losses, **metrics}, 'new_vars': new_vars}
        step_rng, rng = random.split(rng)
        val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=0)
        (loss, stats), grads = val_and_grad(train_state.get_params(), step_rng)
        new_train_state = train_state.apply_gradients(grads, stats['new_vars'])
        return new_train_state, rng, stats['metrics']
    
    def compute_loss(train_state, params, rng, *data, train=True):
        pass
    
    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, *data):
        losses, metrics, aux, new_vars = self.compute_loss(train_state,
                                                           train_state.get_params(),
                                                           rng, *data, train=False)
        metrics = {'loss': sum(losses.values()), **losses, **metrics}
        return metrics, aux


@dataclass
class BaseVAELearner(Learner):
    cond_size : int
    input_size : int
    input_shape : Sequence[int]
    output_size : int
    model_name : str = 'vae'
    latent_dim : int = 10
    learning_rate : float = 1e-4
    lr_decay : float = 0.1
    beta1 : float = 0.5
    beta2 : float = 0.9
    beta : float = 0.5
    num_enc_layers : int = 2
    num_dec_layers : int = 2
    enc_hidden_size : float = 0.5
    dec_hidden_size : float = 0.5
    submodules : 'dict[str, Module]' = super.submodules
    submodule = submodule_decorator(submodules)
    
    @submodule
    def encoder(self):
        enc_hid = int(np.floor(self.enc_hidden_size * self.input_size))
        encoder = Encoder(input_size=self.input_size, 
                          hidden_sizes=(self.num_enc_layers - 1) * (enc_hid,),
                          output_size=enc_hid,
                          latent_dim=self.latent_dim)
        return encoder
    
    @submodule
    def decoder(self):
        dec_hid = int(np.floor(self.dec_hidden_size * self.latent_dim))
        decoder = MLP(input_size=self.latent_dim,
                      hidden_sizes=(self.num_dec_layers - 1) * (dec_hid,),
                      output_size=self.input_size,
                      activation=nn.log_sigmoid)
        return decoder
        
        
    # commented to prevent JAX tracer leaks
    #@partial(jit, static_argnums=0)
    def initial_state(self, total_steps, rng):
        rngs = random.split(rng, len(self.submodules.items()))
        optim = lambda: adam(decay(self.learning_rate, total_steps, self.lr_decay), self.beta1, self.beta2)
        train_state = LearnerState()
        for (name, module), rng in zip(self.submodules.items(), rngs):
            train_state = train_state.add_module(name, module, module.input_shape, optim(), rng)
        return train_state
        
    def compute_loss(self, train_state, params, rng, *data, train=True):
        inputs, conds = data
        mu, log_sigma = train_state.apply_module('encoder', params, inputs)
        r = random.normal(rng, (mu.shape[0], self.latent_dim))
        codes = mu + r * jnp.exp(log_sigma)
        codes = self.concat_conds(codes, conds)
        reconst = train_state.apply_module('decoder', params, inputs)
        reconst = reconst.reshape((-1, *self.input_shape))
        aux = {'image': inputs, 'label': conds, 'output': jnp.exp(reconst)}
        losses = {'reconst_loss': bce_loss(inputs, reconst), 
                  'penalty_kl_loss': kl_loss(mu, log_sigma)}
        return losses, {}, aux, {}
    
    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, dec_state, dec_params, rng, conds):
        codes = random.normal(rng, (n, self.latent_dim))
        codes = self.concat_conds(codes, conds)
        reconst = dec_state.apply_fn(dec_params, codes)
        return reconst
    
    @partial(jit, static_argnums=0)
    def concat_conds(self, codes, conds):
        return codes


class BaseCondVAELearner(BaseVAELearner):
    model_name : str = 'class_vae'
        
    def make_decoder(self):
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.output_size))
        decoder =  MLP(input_size=self.latent_dim + self.cond_size, 
                       hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,),
                       output_size=self.output_size,
                       activation=nn.log_sigmoid)
        return decoder, (self.latent_dim + self.cond_size,)
    
    @partial(jit, static_argnums=0)
    def concat_conds(self, codes, conds):
        return jnp.concatenate([codes, one_hot(conds, self.cond_size)], axis=-1)
    

class BaseGSVAELearner(BaseVAELearner):
    model_name : str = 'gsvae'
    max_temp : float = 0.5
    temp_rate : float = 0.1
    temp_interval : int = 2
    num_values : int = 10

    def make_encoder(self):
        enc_hidden_size = int(np.floor(self.enc_hidden_size * self.input_size))
        return MLP(input_size=self.input_size,
                   hidden_sizes=(self.num_enc_layers - 1) * (enc_hidden_size,),
                   output_size=self.latent_dim * self.num_values)
        
    def make_decoder(self):
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.input_size))
        decoder = MLP(input_size=self.latent_dim * self.num_values, 
                      hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,),
                      output_size=self.output_size,
                      activation=nn.log_sigmoid)

    @partial(jit, static_argnums=0)
    def initial_state(self, epochs, rng):
        enc_rng, dec_rng = random.split(rng)
        encoder, decoder = self.make_encoder(), self.make_decoder()
        enc_params = encoder.init(enc_rng, jnp.ones((10, self.input_size), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.num_values * self.latent_dim), jnp.float32))
        make_optim = lambda: adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return GSVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(), 
                                      self.temp_interval, self.max_temp, self.temp_rate)
        
    def compute_loss(self, train_state, enc_params, dec_params, rng, *data, train=True):
        inputs, conds = data
        logprobs = train_state.enc_state.apply_fn(enc_params, inputs)
        logprobs = logprobs.reshape((-1, self.latent_dim, self.num_values))
        g = -jnp.log(-jnp.log(random.uniform(rng, logprobs.shape) + 1e-20) + 1e-20)
        train_sample = lambda x: nn.softmax((x + g)/train_state.temp, axis=-1)
        test_sample = lambda x: one_hot(jnp.argmax(x, axis=-1), self.num_values)
        codes = lax.cond(train, train_sample, test_sample, logprobs)
        codes = codes.reshape((codes.shape[0], -1))
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        reconst = reconst.reshape((-1, *self.input_shape))
        aux = {'image': inputs, 'label': conds, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), 0, aux
    
    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, dec_state, dec_params, rng, conds):
        codes = random.randint(rng, (n, self.latent_dim), 0, self.num_values)
        codes = one_hot(codes, self.num_values).reshape((codes.shape[0], -1))
        reconst = dec_state.apply_fn(dec_params, codes)
        return reconst


class BaseCondGSVAELearner(BaseGSVAELearner, BaseCondVAELearner):
    model_name : str = 'class_gsvae'

    @partial(jit, static_argnums=0)
    def initial_state(self, epochs, rng):
        enc_rng, dec_rng = random.split(rng)
        dec_input_size = self.latent_dim * self.num_values + self.cond_size
        enc_hidden_size = int(np.floor(self.enc_hidden_size * self.input_size))
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.input_size))
        encoder = MLP(input_size=self.input_size,
                      hidden_sizes=(self.num_enc_layers - 1) * (enc_hidden_size,),
                      output_size=self.latent_dim * self.num_values)
        decoder = MLP(input_size=dec_input_size, 
                      hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,),
                      output_size=self.output_size,
                      activation=nn.log_sigmoid)
        enc_params = encoder.init(enc_rng, jnp.ones((10, self.input_size), jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, dec_input_size), jnp.float32))
        make_optim = lambda: adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return GSVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(), 
                                      self.temp_interval, self.max_temp, self.temp_rate)

    def compute_loss(self, train_state, enc_params, dec_params, rng, *data, train=True):
        inputs, conds = data
        logprobs = train_state.enc_state.apply_fn(enc_params, inputs)
        logprobs = logprobs.reshape((-1, self.latent_dim, self.num_values))
        g = -jnp.log(-jnp.log(random.uniform(rng, logprobs.shape) + 1e-20) + 1e-20)
        train_sample = lambda x: nn.softmax((x + g)/train_state.temp, axis=-1)
        test_sample = lambda x: one_hot(jnp.argmax(x, axis=-1), self.num_values)
        codes = lax.cond(train, train_sample, test_sample, logprobs)
        codes = self.concat_conds(codes, conds)
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        reconst = reconst.reshape((-1, *self.input_shape))
        aux = {'image': inputs, 'label': conds, 'output': jnp.exp(reconst)}
        return bce_loss(inputs, reconst), 0, aux
    
    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, dec_state, dec_params, rng, conds):
        codes = random.randint(rng, (n, self.latent_dim), 0, self.num_values)
        codes = one_hot(codes, self.num_values)
        codes = self.concat_conds(codes, conds)
        reconst = dec_state.apply_fn(dec_params, codes)
        return reconst


@dataclass
class MADELearner:  
    
    input_size : int
    input_shape : Sequence[int]
    cond_size : int
    model_name : str = 'made'
    latent_dim : int = 784
    num_values : int = 2
    cond_size : int = 10
    learning_rate : float = 1e-4
    lr_decay : float = 0.1
    beta1 : float = 0.9
    beta2 : float = 0.5
    num_layers : int = 2
    hidden_size : int = 2
    
    #@partial(jit, static_argnums=0)
    def initial_state(self, epochs, rng):
        size = self.hidden_size * self.latent_dim * self.num_values
        sizes = (self.num_layers - 1) * (int(np.floor(size)),)
        made = MaskedMLP(latent_dim=self.latent_dim, 
                         num_values=self.num_values,
                         hidden_sizes=sizes)
        params = made.init(rng, jnp.ones((10, self.latent_dim, self.num_values), jnp.float32))
        optim = adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return MADETrainState.create(made, params, optim)
    
    def compute_loss(self, train_state, params, inputs, conds):
        logprobs = train_state.made_state.apply_fn(params, inputs)
        return jnp.mean(jnp.sum(-inputs * logprobs, axis=(1, 2)))
    
    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, inputs, conds):
        inputs = one_hot(inputs.reshape((-1, self.latent_dim)), self.num_values)
        conds = one_hot(conds.reshape((-1,)), self.cond_size)
        def loss_fn(params):
            loss = self.compute_loss(train_state, params, inputs, conds)
            return loss, {'loss': loss}
        params = train_state.made_state.params
        (loss, batch_stats), grads = value_and_grad(loss_fn, has_aux=True)(params)
        new_train_state = train_state.apply_gradients(grads=grads)
        return new_train_state, rng, batch_stats
    
    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, made_state, made_params, rng, conds):
        rng = random.split(rng, self.latent_dim)
        samples = jnp.ones((n, self.latent_dim, self.num_values))
        def sample_next_dim(i, samples):
            logprobs = made_state.apply_fn(made_params, samples)
            next_dim = random.categorical(rng[i, :], logprobs[:, i, :], axis=-1)
            return index_update(samples, index[:, i, :], one_hot(next_dim, self.num_values))
        samples = lax.fori_loop(0, self.latent_dim, jit(sample_next_dim), samples)
        return samples
    
    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, inputs, conds, n=25):
        rng, label_rng = random.split(rng)
        inputs = one_hot(inputs.reshape((-1, self.latent_dim)), self.num_values)
        conds = one_hot(conds.reshape((-1,)), self.cond_size)
        loss = self.compute_loss(train_state, train_state.made_state.params, inputs, conds)
        gen_labels = random.randint(label_rng, (n,), 0, self.cond_size)
        generated = self.generate(n, train_state.made_state, train_state.made_state.params, 
                                  rng, one_hot(gen_labels, self.cond_size))
        generated = jnp.argmax(generated, axis=-1).reshape((-1, 28, 28, 1))
        inputs = jnp.argmax(inputs, axis=-1).reshape((-1, 28, 28, 1))
        metrics = {'loss': loss}
        data = {'generated': generated, 
                'image': inputs, 
                'output': inputs, 
                'label': jnp.zeros((inputs.shape[0],)), 
                'generated_label': gen_labels}
        return metrics, data
    
 
class CondMADELearner(MADELearner):  
    
    model_name : str = 'class_made'
    
    #@partial(jit, static_argnums=0)
    def initial_state(self, epochs, rng):
        size = self.hidden_size * self.latent_dim * self.num_values
        sizes = (self.num_layers - 1) * (int(np.floor(size)),)
        made = CondMaskedMLP(latent_dim=self.latent_dim, 
                              num_values=self.num_values,
                              cond_size=self.cond_size,
                              hidden_sizes=sizes)
        inputs = jnp.ones((10, self.latent_dim, self.num_values), jnp.float32)
        classes = jnp.zeros((10, self.cond_size))
        params = made.init(rng, inputs, classes)
        optim = adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return MADETrainState.create(made, params, optim)
    
    def compute_loss(self, train_state, params, inputs, conds):
        logprobs = train_state.made_state.apply_fn(params, inputs, conds)
        return jnp.mean(jnp.sum(-inputs * logprobs, axis=(1, 2)))
    
    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, made_state, made_params, rng, conds):
        rng = random.split(rng, self.latent_dim)
        samples = jnp.ones((n, self.latent_dim, self.num_values))
        def sample_next_dim(i, samples):
            logprobs = made_state.apply_fn(made_params, samples, conds)
            next_dim = random.categorical(rng[i, :], logprobs[:, i, :], axis=-1)
            return index_update(samples, index[:, i, :], one_hot(next_dim, self.num_values))
        samples = lax.fori_loop(0, self.latent_dim, jit(sample_next_dim), samples)
        return samples


@dataclass
class BaseVQVAELearner(BaseVAELearner):
    model_name : str = 'vqvae'
    embedding_dim : int = 10
    num_values : int = 10
    commitment_cost : float = 0.25
    beta : float = 0.5
    ema_vq : bool = False
    vq_momentum : float = 0.9

    def initial_state(self, epochs, rng):
        kwargs = dict(embedding_dim=self.embedding_dim,
                      num_embeddings=self.num_values,
                      commitment_cost=self.commitment_cost)
        if self.ema_vq:
            kwargs['momentum'] = self.vq_momentum
        quantizer = KMeansQuantizer(**kwargs) if self.ema_vq else Quantizer(**kwargs)
        return self._initial_state_helper(quantizer, epochs, rng)
            
    @partial(jit, static_argnums=(0, 1))
    def _initial_state_helper(self, quantizer, epochs, rng):
        enc_rng, dec_rng, vq_rng = random.split(rng, 3)
        enc_hidden_size = int(np.floor(self.enc_hidden_size * self.input_size))
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.input_size))
        encoder = MLP(input_size=self.input_size, 
                      hidden_sizes=(self.num_enc_layers - 1) * (enc_hidden_size,),
                      output_size=self.latent_dim * self.embedding_dim)
        decoder = MLP(input_size=self.latent_dim * self.embedding_dim, 
                      hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,),
                      output_size=self.output_size,
                      activation=nn.log_sigmoid)
        enc_params = encoder.init(enc_rng, jnp.ones((10,) + self.input_shape, jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, self.latent_dim, self.embedding_dim), jnp.float32))
        vq_params = quantizer.init(vq_rng, jnp.ones((10, self.latent_dim, self.embedding_dim), jnp.float32))
        make_optim = lambda: adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return VQVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(),
                                      quantizer, vq_params, make_optim())
        
    def compute_loss(self, train_state, enc_params, dec_params, vq_params, 
                     rng, *data, train=True):
        inputs, conds = data
        latents = train_state.enc_state.apply_fn(enc_params, inputs)
        latents = latents.reshape((-1, self.latent_dim, self.embedding_dim))
        (codes, penalty, vq_aux), new_vars = train_state.vq_state.apply_fn(vq_params, latents, train=train, 
                                                                           mutable=['embedding_vars'])
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        reconst = reconst.reshape((-1, *self.input_shape))
        aux = {'image': inputs, 'label': conds, 'output': jnp.exp(reconst), 'latents': latents}
        return bce_loss(inputs, reconst), penalty, {**aux, **vq_aux, 'new_vars': new_vars}
    
    @partial(jit, static_argnums=0)
    def train_step(self, train_state, rng, *data):
        def loss_fn(enc_params, dec_params, vq_params, rng):
            reconst_loss, penalty, aux = self.compute_loss(train_state, enc_params, dec_params, 
                                                           vq_params, rng, *data, train=True)
            metrics = {'loss': reconst_loss + self.beta * penalty, 
                       'reconst_loss': reconst_loss, 
                       'penalty_loss': penalty,
                       'perplexity': aux['perplexity']}
            return reconst_loss + self.beta * penalty, {'metrics': metrics, 'new_vars': aux['new_vars']}
        
        step_rng, rng = random.split(rng)
        val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))
        (loss, stats), (e_grads, d_grads, q_grads) = val_and_grad(train_state.enc_state.params, 
                                                                 train_state.dec_state.params, 
                                                                 train_state.vq_state.params, 
                                                                 step_rng)
        new_train_state = train_state.apply_gradients(enc_grads=e_grads, 
                                                      dec_grads=d_grads, 
                                                      vq_grads=q_grads,
                                                      new_vars=stats['new_vars'])
        return new_train_state, rng, stats['metrics']
    
    def make_made(self):
        return MADELearner(latent_dim=self.latent_dim,
                           num_values=self.num_values,
                           learning_rate=self.made_learning_rate,
                           lr_decay=self.made_lr_decay,
                           beta1=self.made_beta1,
                           beta2=self.made_beta2,
                           num_layers=self.made_num_layers,
                           hidden_size=self.made_hidden_size)

    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, dec_state, dec_params, made_state, made_params, embeddings, rng, conds):
        samples = self.make_made().generate(n, made_state, made_params, rng, conds)
        idx = jnp.argmax(samples.reshape((-1, self.latent_dim, self.num_values)), -1)
        quantized = device_put(embeddings.T)[(idx,)]
        outputs = dec_state.apply_fn(dec_params, quantized)
        outputs = outputs.reshape((-1, *self.input_shape))
        return outputs
    
    def train_made(self, encodings, labels, rng, epochs):
        made = self.make_made()
        rng, init_rng = random.split(rng)
        state = made.initial_state(init_rng)
        
        steps_per_epoch = encodings.shape[0] // self.made_batch
        epoch_metrics_np = dict()
        for epoch in range(epochs):
            perms = random.permutation(rng, encodings.shape[0])
            perms = perms[:steps_per_epoch * self.made_batch]
            perms = perms.reshape((steps_per_epoch, self.made_batch))
            batch_metrics = list()
            for perm in tqdm(perms, leave=False):
                enc_batch = encodings[perm, ...]
                label_batch = labels[perm, ...]
                state, rng, metrics = made.train_step(state, rng, enc_batch, label_batch)
                batch_metrics.append(metrics)
            batch_metrics_np = device_get(batch_metrics)
            epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                                for metric in batch_metrics_np[0]}
            state = state.next_epoch()
        epoch_metrics_np = {"made_" + metric: value for (metric, value) in epoch_metrics_np.items()}
        return state, epoch_metrics_np
    
    @partial(jit, static_argnums=0)
    def evaluate(self, train_state, rng, *data):
        reconst_loss, kl_penalty, aux = self.compute_loss(train_state,
                                                          train_state.enc_state.params, 
                                                          train_state.dec_state.params,
                                                          rng, *data, train=False)
        metrics = {'loss': reconst_loss + self.beta * kl_penalty, 
                   'reconst_loss': reconst_loss, 
                   'penalty_loss': kl_penalty,
                   'perplexity': aux.pop('perplexity')}
        return metrics, aux
    
    
class BaseCondVQVAELearner(BaseVQVAELearner, BaseCondVAELearner):
    model_name : str = 'class_vqvae'
            
    @partial(jit, static_argnums=(0, 1))
    def _initial_state_helper(self, quantizer, epochs, rng):
        enc_rng, dec_rng, vq_rng = random.split(rng, 3)
        dec_input_size = self.latent_dim * self.embedding_dim + self.cond_size
        enc_hidden_size = int(np.floor(self.enc_hidden_size * self.input_size))
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.input_size))
        encoder = MLP(input_size=self.input_size, 
                      hidden_sizes=(self.num_enc_layers - 1) * (enc_hidden_size,),
                      output_size=self.latent_dim * self.embedding_dim)
        decoder = MLP(input_size=dec_input_size, 
                      hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,),
                      output_size=self.output_size,
                      activation=nn.log_sigmoid)
        enc_params = encoder.init(enc_rng, jnp.ones((10,) + self.input_shape, jnp.float32))
        dec_params = decoder.init(dec_rng, jnp.ones((10, dec_input_size), jnp.float32))
        vq_params = quantizer.init(vq_rng, jnp.ones((10, self.latent_dim, self.embedding_dim), jnp.float32))
        make_optim = lambda: adam(decay(self.learning_rate, epochs, self.lr_decay), self.beta1, self.beta2)
        return VQVAETrainState.create(encoder, enc_params, make_optim(),
                                      decoder, dec_params, make_optim(),
                                      quantizer, vq_params, make_optim())
        
    def compute_loss(self, train_state, enc_params, dec_params, vq_params, rng, *data, train=True):
        inputs, conds = data
        latents = train_state.enc_state.apply_fn(enc_params, inputs)
        latents = latents.reshape((-1, self.latent_dim, self.embedding_dim))
        (codes, penalty, vq_aux), new_vars = train_state.vq_state.apply_fn(vq_params, latents, train=train, 
                                                                           mutable=['embedding_vars'])
        codes = self.concat_conds(codes, conds)
        reconst = train_state.dec_state.apply_fn(dec_params, codes)
        reconst = reconst.reshape((-1, *self.input_shape))
        aux = {'image': inputs, 'label': conds, 'output': jnp.exp(reconst), 'latents': latents}
        return bce_loss(inputs, reconst), penalty, {**aux, **vq_aux, 'new_vars': new_vars}
    
    def make_made(self):
        return CondMADELearner(latent_dim=self.latent_dim,
                                num_values=self.num_values,
                                cond_size=self.cond_size,
                                learning_rate=self.made_learning_rate,
                                lr_decay=self.made_lr_decay,
                                beta1=self.made_beta1,
                                beta2=self.made_beta2,
                                num_layers=self.made_num_layers,
                                hidden_size=self.made_hidden_size)

    @partial(jit, static_argnums=(0, 1))
    def generate(self, n, dec_state, dec_params, made_state, made_params, embeddings, rng, conds):
        samples = self.make_made().generate(made_state, made_params, rng, conds)
        idx = jnp.argmax(samples.reshape((-1, self.latent_dim, self.num_values)), -1)
        quantized = device_put(embeddings.T)[(idx,)]
        quantized = self.concat_conds(quantized, conds)
        outputs = dec_state.apply_fn(dec_params, quantized)
        return outputs






