from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax.struct import dataclass
from jax import lax, numpy as jnp

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
class LearnerState:
    states : 'dict[str, TrainState]' = {}
    epoch : int = 0
    
    def apply_module(self, name, params, *x):
        return self.states[name].apply_fn(params[name], *x)
    
    def add_module(self, name, module, input_shape, optim, rng):
        params = module.init(rng, jnp.ones((10,) + input_shape, jnp.float32))
        new_state = TrainState.create(apply_fn=module.apply, 
                                      params=params, 
                                      tx=optim, 
                                      batch_stats={})
        return LearnerState(states={name: new_state, **self.states}, epoch=0)
    
    def get_params(self):
        return {name: state.params for name, state in self.states.items()}
    
    def next_epoch(self):
        return LearnerState(states=self.states, epoch=self.epoch + 1)
    
    def apply_gradients(self, grads, new_vars):
        new_states = {name: state.apply_gradients(grads=grads[name]) 
                      for name, state in self.states.items()}
        new_states = {name: self.states[name].replace(
                                params=self.states[name].params.copy(add_or_replace=vars))
                      for name, vars in new_vars.items()}
        return LearnerState(states=new_states, epoch=self.epoch)
    
    
@dataclass
class GSVAELearnerState(LearnerState):
    temp : float
    temp_interval : int
    max_temp : float
    temp_rate : float
    
    @classmethod
    def create(cls, modules, params, optims,
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
    
    @classmethod
    def create(cls, enc_module, enc_params, enc_optim,
               dec_module, dec_params, dec_optim, 
               vq_module, vq_params, vq_optim):
        vae_train_state = VAETrainState.create(enc_module, enc_params, enc_optim,
                                               dec_module, dec_params, dec_optim)
        vq_state = TrainState.create(apply_fn=vq_module.apply,
                                     params=vq_params,
                                     tx=vq_optim,
                                     batch_stats={})
        return VQVAETrainState(enc_state=vae_train_state.enc_state,
                               dec_state=vae_train_state.dec_state,
                               vq_state=vq_state, 
                               epoch=0)

    def next_epoch(self):
        return VQVAETrainState(enc_state=self.enc_state, 
                               dec_state=self.dec_state, 
                               vq_state=self.vq_state,
                               epoch=self.epoch + 1)
    
    def apply_gradients(self, enc_grads, dec_grads, vq_grads, new_vars={}):
        enc_state = self.enc_state.apply_gradients(grads=enc_grads)
        dec_state = self.dec_state.apply_gradients(grads=dec_grads)
        vq_state = self.vq_state.apply_gradients(grads=vq_grads)
        new_vars = vq_state.params.copy(add_or_replace=new_vars)
        vq_state = vq_state.replace(params=new_vars)
        return VQVAETrainState(enc_state=enc_state, 
                               dec_state=dec_state, 
                               vq_state=vq_state, 
                               epoch=self.epoch)