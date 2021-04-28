import jax
import numpy as np
from jax import numpy as jnp
from jax import jit
from flax import linen as nn
from flax import optim
from jax import grad
from jax import random
from jax.experimental.host_callback import id_print
import tensorflow_datasets as tfds
from absl import logging

def get_datasets():
    builder = tfds.builder('mnist')
    builder.download_and_prepare()
    train = tfds.as_numpy(builder.as_dataset(split='train', batch_size=-1))
    test = tfds.as_numpy(builder.as_dataset(split='test', batch_size=-1))
    train['image'] = jnp.float32(train['image']) / 255.
    test['image'] = jnp.float32(test['image']) / 255.
    return train, test

'''class MLP(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        for size in hidden_sizes:
            x = nn.relu(nn.Dense(features=size)(x))
        x = nn.softmax(nn.Dense(featuers=10)(x))
        return x
'''
class ConvNet(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(features=64, kernel_size=(2, 2))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.log_softmax(nn.Dense(10)(x))

        return x

    def compute_loss(self, logits, label):
        one_hot = (label[..., None] == jnp.arange(10)[None]).astype(jnp.float32)
        return -jnp.mean(jnp.sum(one_hot * logits, axis=-1))

    def compute_metrics(self, logits, label):
        metrics = {'loss': self.compute_loss(logits, label)}
        metrics['accuracy'] = jnp.mean(jnp.argmax(logits, -1) == label)
        return metrics

    def train_epoch(self, optimizer, images, labels, batch_size, epoch, rng):
        def train_step(optimizer, image, label):
            def loss_func(params):
                logits = self.apply({'params': params}, image)
                return self.compute_loss(logits, label), logits
            (loss, logits), grad = jax.value_and_grad(loss_func, has_aux=True)(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            return self.compute_metrics(logits, label), optimizer
        train_step = jit(train_step)
        steps_per_epoch = images.shape[0] // batch_size
        perms = jax.random.permutation(rng, images.shape[0])
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        batch_metrics = list()
        for perm in perms:
            image_batch = images[perm, ...]
            label_batch = labels[perm, ...]
            metrics, optimizer = train_step(optimizer, image_batch, label_batch)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                            for metric in batch_metrics_np[0]}
        logging.info('train_epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                     epoch_metrics_np['loss'], epoch_metrics_np['accuracy'])
        return optimizer, epoch_metrics_np

    def eval_model(self, params, test_images, test_labels):
        def evaluate(params, image, label):
            logits = self.apply({'params': params}, test_images)
            return self.compute_metrics(logits, test_labels)
        metrics = jit(evaluate)(params, test_images, test_labels)
        summary = jax.tree_map(lambda x: x.item(), jax.device_get(metrics))
        return summary['loss'], summary['accuracy']

    def train(self, optimizer, batch_size, epochs, train, test=None):
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
        init_params = self.init(init_rng, init_val)['params']
        optimizer = optimizer.create(init_params)
        for epoch in range(1, epochs + 1):
            rng, input_rng = jax.random.split(rng)
            optimizer, train_metrics = self.train_epoch(optimizer, train['image'], train['label'], 
                                                        batch_size, epoch, input_rng)
            if test:
                loss, accuracy = self.eval_model(optimizer.target, test['image'], test['label'])
                print("test_epoch: {0}, loss: {1}, accuracy: {2}".format(epoch, loss, accuracy))
        return self


class Autoencoder(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(features=64, kernel_size=(2, 2))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(256)(x))
        codes = nn.log_softmax(nn.Dense(10)(x))
        x = nn.relu(nn.Dense(256)(codes))
        x = nn.relu(nn.Dense(784)(x))
        x = x.reshape((x.shape[0], 28, 28, -1))
        return x, codes

    def compute_loss(self, reconst, image):
        return jnp.mean(jnp.sum((reconst - image) * (reconst - image), axis=-1))

    def train_epoch(self, optimizer, images, labels, batch_size, epoch, rng):
        def train_step(optimizer, image, label):
            def loss_func(params):
                reconst, codes = self.apply({'params': params}, image)
                return self.compute_loss(reconst, image), codes
            (loss, codes), grad = jax.value_and_grad(loss_func, has_aux=True)(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            return {'loss': loss}, optimizer
        train_step = jit(train_step)
        steps_per_epoch = images.shape[0] // batch_size
        perms = jax.random.permutation(rng, images.shape[0])
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        batch_metrics = list()
        for perm in perms:
            image_batch = images[perm, ...]
            label_batch = labels[perm, ...]
            metrics, optimizer = train_step(optimizer, image_batch, label_batch)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                            for metric in batch_metrics_np[0]}
        logging.info('train_epoch: %d, loss: %.4f', epoch, epoch_metrics_np['loss'])
        return optimizer, epoch_metrics_np

    def eval_model(self, params, test_images, test_labels):
        def evaluate(params, image, label):
            reconst, codes = self.apply({'params': params}, test_images)
            return {'loss': self.compute_loss(reconst, test_images)}
        metrics = jit(evaluate)(params, test_images, test_labels)
        summary = jax.tree_map(lambda x: x.item(), jax.device_get(metrics))
        return summary['loss']

    def train(self, optimizer, batch_size, epochs, train, test=None):
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
        init_params = self.init(init_rng, init_val)['params']
        optimizer = optimizer.create(init_params)
        for epoch in range(1, epochs + 1):
            rng, input_rng = jax.random.split(rng)
            optimizer, train_metrics = self.train_epoch(optimizer, train['image'], train['label'], 
                                                        batch_size, epoch, input_rng)
            if test:
                loss = self.eval_model(optimizer.target, test['image'], test['label'])
                print("test_epoch: {0}, loss: {1}".format(epoch, loss))
        return self


class VariationalAutoencoder(nn.Module):
    
    @nn.compact
    def __call__(self, x, rng=random.PRNGKey(0)):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(nn.Conv(features=64, kernel_size=(2, 2))(x))
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(20)(nn.relu(nn.Dense(256)(x)))
        r = random.normal(rng, (x.shape[0], 10))
        mu = x[:, :10]
        log_sigma = x[:, 10:]
        x = mu + r * jnp.exp(log_sigma)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(784)(x))
        x = x.reshape((x.shape[0], 28, 28, -1))
        return x, mu, log_sigma

    def compute_loss(self, reconst, image, mu, log_sigma):
        reconst_loss = jnp.mean(np.sum((reconst - image) * (reconst - image), axis=-1))
        kl_loss = 0#.5 * jnp.sum(1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma))
        return reconst_loss + kl_loss

    def train_epoch(self, optimizer, images, labels, batch_size, epoch, rng):
        def train_step(optimizer, image, label, step_rng):
            def loss_func(params):
                reconst, mu, log_sigma = self.apply({'params': params}, image, step_rng)
                return self.compute_loss(reconst, image, mu, log_sigma), (mu, log_sigma)
            (loss, (mu, log_sigma)), grad = jax.value_and_grad(loss_func, has_aux=True)(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            return {'loss': loss}, optimizer
        train_step = jit(train_step)
        steps_per_epoch = images.shape[0] // batch_size
        perms = jax.random.permutation(rng, images.shape[0])
        perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
        batch_metrics = list()
        for perm in perms:
            image_batch = images[perm, ...]
            label_batch = labels[perm, ...]
            rng, step_rng = random.split(rng)
            metrics, optimizer = train_step(optimizer, image_batch, label_batch, step_rng)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {metric: np.mean([metrics[metric] for metrics in batch_metrics_np])
                            for metric in batch_metrics_np[0]}
        logging.info('train_epoch: %d, loss: %.4f', epoch, epoch_metrics_np['loss'])
        return optimizer, epoch_metrics_np

    def eval_model(self, params, test_images, test_labels):
        def evaluate(params, image, label):
            reconst, mu, log_sigma = self.apply({'params': params}, test_images)
            return {'loss': self.compute_loss(reconst, test_images, mu, log_sigma)}
        metrics = jit(evaluate)(params, test_images, test_labels)
        summary = jax.tree_map(lambda x: x.item(), jax.device_get(metrics))
        return summary['loss']

    def train(self, optimizer, batch_size, epochs, train, test=None):
        rng = random.PRNGKey(0)
        rng, init_rng = random.split(rng)
        init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
        init_params = self.init(init_rng, init_val)['params']
        optimizer = optimizer.create(init_params)
        for epoch in range(1, epochs + 1):
            rng, input_rng = random.split(rng)
            optimizer, train_metrics = self.train_epoch(optimizer, train['image'], train['label'], 
                                                        batch_size, epoch, input_rng)
            if test:
                loss = self.eval_model(optimizer.target, test['image'], test['label'])
                print("test_epoch: {0}, loss: {1}".format(epoch, loss))
        return self
        

learning_rate = 0.01
momentum = 0.99
epochs = 10
batch_size = 60
coder = VariationalAutoencoder().train(optim.Momentum(learning_rate, momentum), batch_size, epochs, *get_datasets())
