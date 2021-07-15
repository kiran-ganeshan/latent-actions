import jax
from jax import random, numpy as jnp
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
import gym
import d4rl
import scipy
from tqdm import tqdm

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

def one_hot(label, num_labels=10):
    return (label[..., None] == jnp.arange(num_labels)[None]).astype(jnp.float32)

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

def write_summary(metrics, train=False, made_coeff=None):
    wandb_metrics = dict()
    if 'made_loss' in metrics and not made_coeff is None:
        agg_metrics = {'loss': metrics['loss'] + metrics['made_loss']}
        other_metrics = {metric: value for metric, value in metrics.items() if metric != 'loss'}
        metrics = {**agg_metrics, **other_metrics}
    for metric, value in metrics.items():
        metric_str = ("train" if train else "test") + "_" + metric
        wandb_metrics[metric_str] = value
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