from models import *


class VAELearner(BaseVAELearner):
                        
    def evaluate(self, train_state, rng, *data, n=25):
        rng, label_rng, generate_rng = random.split(rng)
        metrics, aux = super(BaseVAELearner).evaluate(train_state, rng, *data)
        conds = random.randint(label_rng, (n,), 0, self.cond_size)
        dec_params = train_state.dec_state.params
        generated = self.generate(n, train_state.dec_state, dec_params, generate_rng, conds)
        gen_data = {'generated': jnp.exp(generated.reshape((-1, *self.input_shape))),
                    'generated_label': conds}
        return metrics, {**gen_data, **aux}


class CondVAELearner(VAELearner, BaseCondVAELearner):
    pass
    
    
class GSVAELearner(BaseGSVAELearner, VAELearner):
    pass
    
    
class CondGSVAELearner(BaseCondGSVAELearner, VAELearner):
    pass
    
    
class VQVAELearner(BaseVQVAELearner):
            
    def evaluate(self, train_state, rng, *data, n=25):
        loss_rng, made_rng, label_rng, generate_rng = random.split(rng, 4)
        metrics, aux = super().evaluate(train_state, loss_rng, *data)
        encodings = aux['encoding_index'].reshape((-1, self.latent_dim))
        made_state, made_metrics = self.train_made(encodings, data[1], made_rng, self.made_epochs)
        embed_col = 'embedding_vars' if self.ema_vq else 'params'
        embeddings = train_state.vq_state.params[embed_col]['embeddings']
        conds = random.randint(label_rng, (n,), 0, self.cond_size)
        generated = self.generate(n, train_state.dec_state, train_state.dec_state.params,
                                  made_state, made_state.params, embeddings, generate_rng, conds)
        metrics = {**metrics, **made_metrics}
        data = {'generated': jnp.exp(generated),
                'generated_label': conds, **aux}
        return metrics, data
    
    
class CondVQVAELearner(BaseCondVQVAELearner, VQVAELearner):
    pass