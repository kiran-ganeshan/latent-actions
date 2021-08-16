import jax
from jax import random, numpy as jnp, jit, partial, value_and_grad
from jax.nn import log_sigmoid, one_hot, sigmoid
from flax.struct import dataclass
from dataclasses import dataclass as dc
from flax.training.train_state import *
from flax.core.frozen_dict import FrozenDict
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
import gym
import d4rl
from typing import Callable, Sequence
from tqdm import tqdm
from modules import MLP, Encoder
from copy import deepcopy
from math import exp, log, floor, pow
from optax import adam

matplotlib.use('agg')

def kl_loss(mu, log_sigma):
    kl_exp = 1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma)
    return -jnp.mean(jnp.sum(kl_exp, axis=-1))

def bce_loss(inputs, outputs):
    bce_exp = inputs * outputs + (1 - inputs) * jnp.log(-jnp.expm1(outputs) + 1e-20)
    return -jnp.mean(jnp.sum(bce_exp, axis=tuple(range(1, bce_exp.ndim))))

def mse_loss(inputs, outputs):
    mse_exp = (inputs - outputs) ** 2
    #return jnp.mean(jnp.sum(mse_exp, axis=tuple(range(1, mse_exp.ndim))))
    return jnp.mean(mse_exp)

def get_mnist_datasets(binarized=True):
    builder = tfds.builder('binarized_mnist' if binarized else 'mnist')
    builder.download_and_prepare()
    train = tfds.as_numpy(builder.as_dataset(split='train', batch_size=-1))
    test = tfds.as_numpy(builder.as_dataset(split='test', batch_size=-1))
    train['image'] = jnp.float32(train['image'])
    test['image'] = jnp.float32(test['image'])
    if binarized:
        train['label'] = jnp.zeros(50000)
        test['label'] = jnp.zeros(10000)
    else:
        train['image'] = train['image'] / 255.
        test['image'] = test['image'] / 255.
    return train, test

def get_rl_datasets(env):
    env = gym.make(env)
    data = d4rl.qlearning_dataset(env)    
    return data, data

def train(coder, epochs, batch_size, test_interval, train_ds, test_ds, seed, made_coeff=None):
    for train_obj, test_obj in zip(train_ds, test_ds):
        assert train_obj.shape[0] == train_ds[0].shape[0]
        assert test_obj.shape[0] == test_ds[0].shape[0]
    rng, init_rng = random.split(random.PRNGKey(seed))
    steps_per_epoch = train_ds[0].shape[0] // batch_size
    state = coder.initial_state(epochs * steps_per_epoch, init_rng)
    for epoch in range(epochs):
        rng, step_rng, shuffle_rng = random.split(rng, 3)
        perms = jax.random.permutation(shuffle_rng, train_ds[0].shape[0])
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        batch_metrics = list()
        for perm in tqdm(perms):
            batch_data = tuple([obj[perm, ...] for obj in train_ds])
            state, rng, metrics = coder.train_step(state, step_rng, *batch_data)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                            for metric in batch_metrics_np[0]}
        write_summary(epoch_metrics_np, True)
        if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
            rng, eval_rng = random.split(rng)
            batch_data = tuple([obj[perm, ...] for obj in test_ds])
            eval_metrics, eval_data = coder.evaluate(state, eval_rng, *batch_data)
            eval_metrics_np, eval_data_np = jax.device_get(eval_metrics), jax.device_get(eval_data)
            write_summary(eval_metrics_np, False, made_coeff)
            write_data(eval_data_np, False)
        state = state.next_epoch()

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    width, height = figure.get_size_inches() * figure.get_dpi()
    figure.tight_layout()
    figure.canvas.draw()
    im = np.fromstring(figure.canvas.tostring_rgb(), dtype='uint8')
    im = im.reshape(int(height), int(width), 3)
    plt.close(figure)
    return im

def image_grid(n : int, 
               image, 
               output, 
               label, 
               feed_forward=True, 
               worst=True, 
               generated=None):
    """Return a nxn grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    for obj in [image, output, label]:
        assert isinstance(obj, np.ndarray)
        assert obj.shape[0] >= n * n
    if generated is not None:
        assert isinstance(generated, np.ndarray)
        assert generated.shape[0] >= n * n
    assert len(label.shape) == 1
    if feed_forward and generated is None:
        output = np.argmax(output, axis=1)
        assert len(output.shape) == 1
        worst_image_it = np.squeeze(np.argwhere(output != label))[-n*n:]
    elif generated is None:
        assert output.shape == image.shape
        output = np.exp(output)
        comparison = jnp.concatenate([image, output], axis=2)
        axes = tuple(range(1, len(output.shape)))
        diff = np.sum((output - image) * (output - image), axis=axes)
        worst_image_it = np.argsort(diff)[-n*n:]
    figure = plt.figure(figsize=(1.6 * n, 1.6 * n))
    use_worst = worst and generated is None
    for (i, index) in enumerate(worst_image_it if use_worst else range(n * n)):
        # Start next subplot.
        if feed_forward and generated is None:
            title = "Predicted: {0}\nActual: {1}".format(output[index], label[index])
        else:
            title = "Label: {}".format(label[index])
        plt.subplot(n, n, i + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if feed_forward and generated is None:
            im = image
        elif generated is None:
            im = comparison
        else:
            im = generated
        plt.imshow(im[index, ...], cmap=plt.cm.binary)
    return plot_to_image(figure)

def histogram(densities):
    densities = densities.reshape((-1, densities.shape[-1]))
    n = np.ceil(np.sqrt(densities.shape[0])).astype(np.intc)
    figure = plt.figure(figsize=(1.6 * n, 1.6 * n))
    for i in range(densities.shape[0]):
        plt.subplot(n, n, i + 1, title="Latent {}".format(i))
        plt.ylim(0, 1)
        plt.grid(False)
        plt.bar(range(1, densities.shape[1] + 1), densities[i, :])
    return plot_to_image(figure)

def scatter_plot(points, centers):
    points = points.reshape((-1, 2))
    centers = centers.reshape((-1, 2))
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=5, c='red')
    plt.scatter(centers[:, 0], centers[:, 1], s=50, c='blue')
    return plot_to_image(figure)

def write_summary(metrics, train=False, log_wandb=True, made_coeff=None):
    wandb_metrics = dict()
    if 'made_loss' in metrics and not made_coeff is None:
        agg_metrics = {'loss': metrics['loss'] + metrics['made_loss']}
        other_metrics = {metric: value for metric, value in metrics.items() if metric != 'loss'}
        metrics = {**agg_metrics, **other_metrics}
    for metric, value in metrics.items():
        metric_str = ("train" if train else "test") + "_" + metric
        wandb_metrics[metric_str] = value
    if log_wandb:
        wandb.log(wandb_metrics)
    summary_str = ("Train " if train else "Test ") + "Results\t"
    for metric, value in metrics.items():
        summary_str += "{0}: {1:.4f}, ".format(metric, value)
    print(summary_str[:-2])

def write_data(data, train=False):
    wandb_img_str = "train_" if train else "test_"
    wandb_metrics = dict()
    if 'image' in data.keys():
        assert 'output' in data.keys() and 'label' in data.keys()
        test_data = (data['image'], data['output'], data['label'])
        grid = image_grid(5, *test_data, False, False)
        wandb_metrics[wandb_img_str + "sample"] = wandb.Image(grid)
        grid = image_grid(5, *test_data, False, True)
        wandb_metrics[wandb_img_str + "worst"] = wandb.Image(grid)
    if 'generated' in data.keys():
        test_data = (data['image'], data['output'], data['generated_label'])
        grid = image_grid(5, *test_data, generated=data['generated'])
        wandb_metrics[wandb_img_str + "generated"] = wandb.Image(grid)
    if 'avg_probs' in data.keys():
        hist = histogram(data['avg_probs'])
        wandb_metrics[wandb_img_str + "avg_probs"] = wandb.Image(hist)
    if 'latents' in data.keys() and data['latents'].shape[-1] == 2:
        assert 'centers' in data.keys() and data['centers'].shape[-1] == 2
        scatter = scatter_plot(data['latents'], data['centers'])
        wandb_metrics[wandb_img_str + "embeddings"] = wandb.Image(scatter)
    wandb.log(wandb_metrics)

def module(method):
    method.is_submodule = True
    return method

def learner(cls):
    modules = getattr(cls, "submodule_funcs", {})
    optims = getattr(cls, "optimizers", {})
    for f in cls.__dict__.values():
        if isinstance(f, Callable) and getattr(f, "is_submodule", False):
            modules[f.__name__] = f
    cls._submodule_outputs = lambda self: {name: f(self) for name, f in modules.items()}
    cls = dataclass(cls)
    
    @dataclass
    class Wrapper(cls):
        
        @staticmethod
        def __new__(self, rng, *args, **kwargs):
            model = cls(*args, **kwargs)
            params, optims = dict(), dict()
            rngs = random.split(rng, len(cls._submodule_outputs(self)))
            for i, (name, tup) in enumerate(cls._submodule_outputs(self).items()):
                print(f"initializing {name}")
                module, optim, input_shape = tup
                params[name] = module.init(rngs[i], jnp.zeros(input_shape, jnp.float32))
                optims[name] = optim
                is_mutable = lambda name: module.bind(params).is_mutable_collection(name)
                mutable = [col for col in params.keys() if is_mutable(col)]
                apply = lambda params, *x: module.apply(params[name], *x)
                print(f"apply is {apply}")
                object.__setattr__(model, name, apply)
                print(f"model.{name} is {getattr(model, name)}")
            return model, MultiTrainState.create(params=params, tx=optims)
        
    return Wrapper
    
TrainState

class MultiTrainState(struct.PyTreeNode):
    """Simpler train state. Doesn't store an apply, leaves that to the learner.

    Synopsis:

    state = TrainState.create(
        params=variables['params'],
        tx=tx)
    grad_fn = jax.grad(make_loss_fn(state.apply_fn))
    for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Attributes:
    step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
    """
    step: int
    params: core.FrozenDict[str, Any]
    tx: core.FrozenDict[str, optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: core.FrozenDict[str, optax.OptState]

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        new_params = self.params
        new_opt_state = self.opt_state
        for name, grads in grads.items():
            updates, new_opt = self.tx[name].update(
                grads, self.opt_state[name], self.params[name])
            new_params = optax.apply_updates(new_params, updates)
            new_opt_state = new_opt_state.replace(name=new_opt)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = dict()
        for name in tx.keys():
            opt_state[name] = tx[name].init(params[name])
        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
    

@learner
class Classifier:
    input_size : int = 784
    num_labels : int = 10
    
    @module
    def classifier(self):
        encoder = MLP(input_size=self.input_size,
                      output_size=self.num_labels,
                      hidden_sizes=(500, 500),
                      activation=log_sigmoid)
        return encoder, (self.input_size,)
        
    def compute_loss(self, params, rng, *data, train=True):
        inputs, labels = data
        preds, params = self.classifier(params, inputs) 
        return {'loss': mse_loss(preds, one_hot(labels, self.num_labels))}, {}, {}, params
    
    def initial_state(self, init_rng):
        module, input_shape, optim = self.classifier()
        params = module.init(init_rng, jnp.zeros(input_shape, jnp.float32))
        return TrainState.create(apply_fn=module.apply, params=params, tx=optim)
    
    
@learner
class Autoencoder:
    input_size : int = 784
    latent_dim : int = 10
    
    @module
    def encoder(self):
        encoder = Encoder(input_size=self.input_size,
                          latent_dim=self.latent_dim,
                          hidden_sizes=(500, 500))
        return encoder, adam(1e-4), (self.input_size,)
    
    @module
    def decoder(self):
        decoder = MLP(input_size=self.latent_dim,
                      output_size=self.input_size,
                      hidden_sizes=(500, 500))
        return decoder, adam(1e-4), (self.latent_dim)
    
    def compute_loss(self, params, rng, image, label, train=False):
        print(image.shape)
        print(f"calling encoder at {self.encoder}")
        print(self.encoder(params, image))
        r = random.normal(rng, mu.shape)
        codes = mu + jnp.exp(log_sigma) * r
        print(f"calling encoder at {self.decoder}")
        output, params = self.decoder(params, codes)
        return output, {'loss': mse_loss(output, image)}, {}, params
    
#@partial(jit, static_argnums=0)
def train_step(coder, train_state, rng, **data):
    def loss_fn(params, rng):
        losses, metrics, aux, new_vars = coder.compute_loss(params, rng, **data, train=True)
        loss = sum(losses.values())
        return loss, {'new_vars': new_vars, 'info': {'loss': loss, **losses, **metrics}}
    step_rng, rng = random.split(rng)
    val_and_grad = value_and_grad(loss_fn, has_aux=True, argnums=0)
    (loss, stats), grads = val_and_grad(train_state.params, step_rng)
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, rng, stats['info']


train_ds, test_ds = get_mnist_datasets(binarized=True)
epochs = 20
batch_size = 60
seed = 0
test_interval = 5
dataset_len = train_ds['image'].shape[0]
rng, init_rng = random.split(random.PRNGKey(seed))
coder, state = Autoencoder(init_rng)
steps_per_epoch = dataset_len // batch_size
for epoch in range(epochs):
    rng, step_rng, shuffle_rng = random.split(rng, 3)
    perms = jax.random.permutation(shuffle_rng, dataset_len)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    batch_metrics = list()
    for perm in tqdm(perms):
        batch_data = {name: obj[perm, ...] for name, obj in train_ds.items()}
        state, rng, metrics = train_step(coder, state, step_rng, **batch_data)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                        for metric in batch_metrics_np[0]}
    write_summary(epoch_metrics_np, True, log_wandb=False)
    if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
        rng, eval_rng = random.split(rng)
        batch_data = tuple([obj[perm, ...] for obj in test_ds])
        eval_metrics, eval_data = coder.evaluate(state, eval_rng, *batch_data)
        eval_metrics_np, eval_data_np = jax.device_get(eval_metrics), jax.device_get(eval_data)
        write_summary(eval_metrics_np, False, log_wandb=False)
        #write_data(eval_data_np, False, log_wandb=False)
    #state = state.next_epoch()
