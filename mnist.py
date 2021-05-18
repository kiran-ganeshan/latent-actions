import jax
import numpy as np
import os, sys
from jax import numpy as jnp
from jax import jit
from flax import linen as nn
from flax import optim
from jax import grad
from jax import random
from jax import lax
from jax.experimental.host_callback import id_print
from tqdm import tqdm
import tensorflow_datasets as tfds
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from flax.linen.initializers import zeros, lecun_normal
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
import io
import datetime
from typing import Sequence
from utils import image_grid, get_datasets, one_hot

FLAGS = flags.FLAGS
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
FLAGS(sys.argv)


class Model(nn.Module):

    model_name : str = 'model'
    
    @nn.compact
    def __call__(self, x, rng, label):
        pass

    def compute_loss(self, x, out, label, **aux):
        pass

    def compute_metrics(self, x, out, label, **aux):
        pass

    def write_summary(summary_writer, metrics, epoch, train=False, images=False):
        pass
     
    def eval_model(self, params, test_images, test_labels, rng):
        def evaluate(params, image, label):
            logits, aux = self.apply({'params': params}, image, rng, label)
            metrics = {**self.compute_metrics(image, logits, label, **aux), **aux,
                       'image': image, 'output': logits, 'label': label}
            return metrics
        metrics = jit(evaluate)(params, test_images, test_labels)
        summary = jax.tree_map(lambda x: np.squeeze(x), jax.device_get(metrics))
        return summary
    
    def train_epoch(self, optimizer, images, labels, batch_size, epoch, rng):
        def train_step(optimizer, image, label, step_rng):
            def loss_func(params):
                out, aux = self.apply({'params': params}, image, rng, label)
                return self.compute_loss(image, out, label, **aux), (out, aux)
            (loss, (out, aux)), grad = jax.value_and_grad(loss_func, has_aux=True)(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            metrics = {**self.compute_metrics(image, out, label, **aux), **aux}
            return metrics, optimizer
        train_step = jit(train_step)
        steps_per_epoch = images.shape[0] // batch_size
        perms = jax.random.permutation(rng, images.shape[0])
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        batch_metrics = list()
        for perm in tqdm(perms):
            image_batch = images[perm, ...]
            label_batch = labels[perm, ...]
            rng, step_rng = random.split(rng)
            metrics, optimizer = train_step(optimizer, image_batch, label_batch, step_rng)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                            for metric in batch_metrics_np[0]}
        return optimizer, epoch_metrics_np

    def train(self, 
              optimizer, 
              batch_size, 
              epochs, 
              train, 
              test=None, 
              test_interval=1,
              save_img_on_train=False,
              save_img_on_test=True):
        # Create new directory for saving info under tmp/{model_name}
        path = os.path.join(FLAGS.save_dir, self.model_name)
        path = os.path.join(path, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        summary_writer = SummaryWriter(path)
        # Initialize model with new random seed
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        init_image = jnp.ones((1, 28, 28, 1), jnp.float32)
        init_label = jnp.ones((1,), jnp.float32)
        init_params = self.init(init_rng, init_image, init_rng, init_label)['params']
        optimizer = optimizer.create(init_params)
        # Train model for {epochs} many epochs
        for epoch in range(1, epochs + 1):
            rng, input_rng = jax.random.split(rng)
            optimizer, train_metrics = self.train_epoch(optimizer, train['image'], train['label'], 
                                                        batch_size, epoch, input_rng)
            self.write_summary(summary_writer, train_metrics, input_rng, epoch, True, save_img_on_train)
            if test and epoch % test_interval == 0:
                summary = self.eval_model(optimizer.target, test['image'], test['label'], input_rng)
                summary_str = "test_epoch: {}".format(epoch)
                for key, val in summary.items():
                    if not isinstance(val, np.ndarray) or not val.shape:
                        summary_str += ", " + key + ": {}".format(val)
                print(summary_str)
                self.write_summary(summary_writer, summary, input_rng, epoch, False, save_img_on_test)
        return self


class FeedForwardModel(Model):
    
    def compute_loss(self, x, out, label, **aux):
        return -jnp.mean(jnp.sum(one_hot(label) * out, axis=-1))

    def compute_metrics(self, x, out, label, **aux):
        metrics = {'loss': self.compute_loss(x, out, label, **aux)}
        metrics['accuracy'] = jnp.mean(jnp.argmax(out, -1) == label)
        return metrics

    def write_summary(self, summary_writer, metrics, rng, epoch, train=False, images=False):
        loss_str = "loss/" + ("train" if train else "test")
        acc_str = "accuracy/" + ("train" if train else "test")
        img_str = "images/" + ("train" if train else "test") + '/epoch{}/'.format(epoch)
        summary_writer.scalar(loss_str, metrics['loss'], epoch)
        summary_writer.scalar(acc_str, metrics['accuracy'], epoch)
        if images:
            data = (metrics['image'], metrics['output'], metrics['label'])
            summary_writer.image(img_str + "sample", image_grid(10, *data, True, False), epoch)
            summary_writer.image(img_str + "worst", image_grid(10, *data, True, True), epoch)


class DensityModel(Model):

    def compute_loss(self, x, out, label, **aux):
        return -jnp.mean(jnp.sum(x * out + (1 - x) * jnp.log(-jnp.expm1(out) + 1e-9), axis=(1, 2, 3)))

    def compute_metrics(self, x, out, label, **aux):
        return {'loss': self.compute_loss(x, out, label, **aux)}

    def write_summary(self, summary_writer, metrics, rng, epoch, train=False, images=False):
        loss_str = "loss/" + ("train" if train else "test")
        img_str = "images/" + ("train" if train else "test") + '/epoch{}/'.format(epoch)
        summary_writer.scalar(loss_str, metrics['loss'], epoch)
        if images:
            data = (metrics['image'], metrics['output'], metrics['label'])
            summary_writer.image(img_str + "sample", image_grid(10, *data, False, False), epoch)
            summary_writer.image(img_str + "worst", image_grid(10, *data, False, True), epoch)

     

class MLP(FeedForwardModel):

    model_name : str = 'mlp'
    hidden_sizes : Sequence[int] = (100,)
    input_size : int = 784
    output_size : int = 10
    output_act : bool = True

    @nn.compact
    def __call__(self, x, rng, label):
        x = x.reshape((-1, self.input_size))
        return self.mlp(x, rng, label), {}

    def mlp(self, x, rng, label):
        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.Dense(features=self.output_size)(x)
        x = lax.cond(self.output_act, nn.log_softmax, lambda x: x, x)
        return x
        

class ClassCondMLP(MLP):
    
    model_name : str = 'class_mlp'

    @nn.compact
    def __call__(self, x, rng, label):
        x = x.reshape((x.shape[0], self.input_size))
        x = jnp.concatenate([x, one_hot(label)], axis=1)
        return self.mlp(x, rng, label), {}


class ConvNet(FeedForwardModel):

    model_name : str = 'conv'
    output_size : int = 10
    output_act : bool = True

    def conv(self, x, rng, label):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(features=64, kernel_size=(2, 2))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

    def mlp(self, x, rng, label):
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(self.output_size)(x)
        x = lax.cond(self.output_act, nn.log_softmax, lambda x: x, x)
        return x

    @nn.compact
    def __call__(self, x, rng, label):
        x = self.conv(x, rng, label)
        x = x.reshape((x.shape[0], -1))
        x = self.mlp(x, rng, label)
        return x, {}


class Autoencoder(DensityModel):

    model_name : str = 'ae'

    def setup(self):
        self.encoder = MLP(hidden_sizes=[500], output_act=False)
        self.decoder = MLP(hidden_sizes=[500], input_size=10, 
                           output_size=784, output_act=False)
    
    @nn.compact
    def __call__(self, x, rng, label):
        codes, en_aux = self.encoder(x, rng, label)
        x, de_aux = self.decoder(codes, rng, label)
        x = nn.log_sigmoid(x.reshape((x.shape[0], 28, 28, -1)))
        return x, {'codes': codes, **en_aux, **de_aux}


class ClassCondAE(Autoencoder):

    model_name : str = 'ccae'

    def setup(self):
        self.encoder = ClassCondMLP(hidden_sizes=[500], output_act=False)
        self.decoder = ClassCondMLP(hidden_sizes=[500], input_size=10,
                                    output_size=784, output_act=False)

class VAE(DensityModel):

    model_name : str = 'vae'
    latent_dim : int = 10
    beta : float = 0.5

    def setup(self):
        self.encoder = MLP(hidden_sizes=[500], 
                           output_size=256, 
                           output_act=False)
        self.decoder = MLP(hidden_sizes=[256, 500], 
                           input_size=self.latent_dim, 
                           output_size=784, 
                           output_act=False)

    @nn.compact
    def __call__(self, x, rng, label):
        x, en_aux = self.encoder(x, rng, label)
        mu = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        log_sigma = nn.Dense(self.latent_dim, kernel_init=zeros)(x)
        r = random.normal(rng, (x.shape[0], self.latent_dim))
        x = mu + r * jnp.exp(log_sigma)
        x, de_aux = self.decoder(x, rng, label)
        x = x.reshape((x.shape[0], 28, 28, -1))
        x = nn.log_sigmoid(x)
        return x, {'mu': mu, 'log_sigma': log_sigma, **en_aux, **de_aux}

    def generate(self, n, rng):
        r = random.normal(rng, (n, self.latent_dim))
        label = random.categorical(rng, jnp.ones(10), shape=(n,))
        x, de_aux = self.decoder(r, rng, label)
        x = x.reshape((x.shape[0], 28, 28, -1))
        x = nn.log_sigmoid(x)
        return x, {'label': label, **de_aux}
    
    def loss_terms(self, x, out, label, **aux):
        assert 'mu' in aux and 'log_sigma' in aux
        mu, log_sigma = aux['mu'], aux['log_sigma']
        reconst_loss = super().compute_loss(x, out, label, **aux)
        kl_exp = 1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma)
        kl_loss = - self.beta * jnp.mean(jnp.sum(kl_exp, axis=-1))
        return reconst_loss, kl_loss

    def compute_loss(self, x, out, label, **aux):
        reconst_loss, kl_loss = self.loss_terms(x, out, label, **aux)
        return reconst_loss + kl_loss

    def compute_metrics(self, x, out, label, **aux):
        reconst_loss, kl_loss = self.loss_terms(x, out, label, **aux)
        return {'loss': reconst_loss + kl_loss, 'penalty_kl_loss': kl_loss, 
                'reconstruction_loss': reconst_loss}

    def write_summary(self, summary_writer, metrics, rng, epochs, train=False, images=False):
        super().write_summary(summary_writer, metrics, rng, epochs, train, images)
        if images:
            gen, aux = self.generate(10, rng)
            img_str = "images/" + ("train" if train else "test") + '/epoch{}/'.format(epochs)
            data = (metrics['image'], metrics['output'], aux['label'])
            summary_writer.image(img_str + "generated", image_grid(10, *data, generated=gen), epochs)
        
def generate_samples(rng, decoder, params):
    keys = ["hidden_sizes", "input_size", "output_size", "output_act"]
    assert all([key in params.keys() for key in keys])
    gen, aux = decoder.apply(params)


class ClassCondVAE(VAE):

    model_name : str = 'ccvae'

    def setup(self):
        self.encoder = ClassCondMLP(hidden_sizes=[500, 256], 
                                    output_size=2 * self.latent_dim, 
                                    output_act=False)
        self.decoder = ClassCondMLP(hidden_sizes=[256, 500], 
                                    input_size=self.latent_dim, 
                                    output_size=784, 
                                    output_act=False)


learning_rate = 0.0001
momentum = 0.99
epochs = 20
batch_size = 60
test_interval = 1

coder = VAE()
coder.train(optim.Momentum(learning_rate, momentum), batch_size, epochs, *get_datasets(), test_interval)
