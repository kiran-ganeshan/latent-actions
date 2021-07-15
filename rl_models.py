from jax import lax, random, jit, partial
from jax import numpy as jnp
from jax.nn import one_hot
from flax import linen as nn
from flax.struct import dataclass
from models import *
from utils import kl_loss, mse_loss



@dataclass
class RLVLearner(BaseCondVAELearner):
    
    model_name : str = 'rl_vae'
    num_evals : int = 10
    gamma : float = 0.99
    output_size : int = 1
    
    def make_decoder(self):
        dec_hidden_size = int(np.floor(self.dec_hidden_size * self.input_size))
        return DoubleQDecoder(input_size=self.latent_dim + self.cond_size, 
                              hidden_sizes=(self.num_dec_layers - 1) * (dec_hidden_size,))
    
    def compute_loss(self, train_state, enc_params, dec_params, rng, *data, train=True):
        actions, obs, rewards, next_obs = data
        fwd_rng, gen_rng = random.split(rng)
        mu, log_sigma = train_state.enc_state.apply_fn(enc_params, actions)
        r = random.normal(fwd_rng, (mu.shape[0], self.latent_dim))
        codes = mu + r * jnp.exp(log_sigma)
        codes = self.concat_conds(codes, obs)
        q = train_state.dec_state.apply_fn(dec_params, codes)
        next_obs = jnp.broadcast_to(next_obs, (self.num_evals, *next_obs.shape))
        next_obs = next_obs.reshape((-1, next_obs.shape[-1]))
        gen_q = self.generate(next_obs.shape[0], train_state.dec_state, 
                              train_state.dec_state.params, gen_rng, next_obs)
        gen_q = jnp.min(gen_q, axis=-1)
        gen_q = gen_q.reshape(self.num_evals, next_obs.shape[0] // self.num_evals)
        target = rewards + self.gamma * jnp.max(gen_q, axis=0)
        target = jnp.expand_dims(target, axis=-1)
        aux = {'q': q, 'action': actions, 'obs': obs, 'reward': rewards}
        return mse_loss(q, target), kl_loss(mu, log_sigma), aux
    
    @partial(jit, static_argnums=0)
    def concat_conds(self, codes, conds):
        return jnp.concatenate([codes, conds], axis=-1)
    
    

class RLGSVLearner(BaseCondGSVAELearner, RLVLearner):
    
    model_name : str = 'rl_gsvae'
    num_evals : int = 10
    gamma : float = 0.99
    
    def compute_loss(self, train_state, enc_params, dec_params, rng, *data, train=True):
        actions, obs, rewards, next_obs = data
        rng, gen_rng = random.split(rng)
        logprobs = train_state.enc_state.apply_fn(enc_params, actions)
        logprobs = logprobs.reshape((-1, self.latent_dim, self.num_values))
        g = -jnp.log(-jnp.log(random.uniform(rng, logprobs.shape) + 1e-20) + 1e-20)
        train_sample = lambda x: nn.softmax((x + g)/train_state.temp, axis=-1)
        test_sample = lambda x: one_hot(jnp.argmax(x, axis=-1), self.num_values)
        codes = lax.cond(train, train_sample, test_sample, logprobs)
        codes = self.concat_conds(codes, obs)
        q = train_state.dec_state.apply_fn(dec_params, codes)
        aux = {'q': q, 'action': actions, 'obs': obs, 'reward': rewards}
        next_obs = next_obs.reshape((-1, next_obs.shape[-1]))
        gen_q = self.generate(next_obs.shape[0], train_state, gen_rng, next_obs)
        gen_q = gen_q.reshape(self.num_evals, *next_obs.shape)
        target = rewards + self.gamma * jnp.max(gen_q, axis=-1)
        return mse_loss(q, target), 0, aux
    
    
class RLVQLearner(BaseCondVQVAELearner, RLVLearner):
    
    model_name : str = 'rl_vqvae'
    num_evals : int = 10
    gamma : float = 0.99
    
    def compute_loss(self, train_state, enc_params, dec_params, vq_params, rng, *data, train=True):
        actions, obs, rewards, next_obs = data
        rng, gen_rng = random.split(rng)
        latents = train_state.enc_state.apply_fn(enc_params, actions)
        latents = latents.reshape((-1, self.latent_dim, self.embedding_dim))
        (codes, penalty, vq_aux), new_vars = train_state.vq_state.apply_fn(vq_params, latents, train=train, 
                                                                           mutable=['embedding_vars'])
        codes = self.concat_conds(codes, obs)
        q = train_state.dec_state.apply_fn(dec_params, codes)
        aux = {'q': q, 'action': actions, 'obs': obs, 'reward': rewards}
        next_obs = next_obs.reshape((-1, next_obs.shape[-1]))
        gen_q = self.generate(next_obs.shape[0], train_state, gen_rng, next_obs)
        gen_q = gen_q.reshape(self.num_evals, *next_obs.shape)
        target = rewards + self.gamma * jnp.max(gen_q, axis=-1)
        return mse_loss(q, target), penalty, {**aux, **vq_aux, 'new_vars': new_vars}
    
    
        
        
        
