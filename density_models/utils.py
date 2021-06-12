from jax import numpy as jnp
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb

matplotlib.use('agg')

def kl_loss(mu, log_sigma):
    kl_exp = 1 + log_sigma - jnp.square(mu) - jnp.exp(log_sigma)
    return -jnp.mean(jnp.sum(kl_exp, axis=-1))

def bce_loss(inputs, outputs):
    bce_exp = inputs * outputs + (1 - inputs) * jnp.log(-jnp.expm1(outputs) + 1e-20)
    return -jnp.mean(jnp.sum(bce_exp, axis=(1, 2, 3)))

def concat_labels(inputs, labels, num_classes=10):
    inputs = inputs.reshape((inputs.shape[0], -1))
    inputs = jnp.concatenate([inputs, one_hot(labels, num_classes)], axis=1)
    return inputs

def get_datasets(binarized=True):
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

def one_hot(label, num_labels=10):
    return (label[..., None] == jnp.arange(num_labels)[None]).astype(jnp.float32)

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

def write_summary(summary_writer, metrics, epochs, train=False, made_coeff=None):
    wandb_metrics = dict()
    if 'made_loss' in metrics and not made_coeff is None:
        agg_metrics = {'loss': metrics['loss'] + metrics['made_loss']}
        other_metrics = {metric: value for metric, value in metrics.items() if metric != 'loss'}
        metrics = {**agg_metrics, **other_metrics}
    for metric, value in metrics.items():
        metric_str = metric + "/" + ("train" if train else "test")
        wandb_metric_str = ("train" if train else "test") + "_" + metric
        summary_writer.scalar(metric_str, value, epochs)
        wandb_metrics[wandb_metric_str] = value
    wandb.log(wandb_metrics)
    summary_str = ("Train " if train else "Test ") + "Results\t"
    for metric, value in metrics.items():
        summary_str += "{0}: {1:.4f}, ".format(metric, value)
    print(summary_str[:-2])

def write_data(summary_writer, data, epochs, train=False):
    img_str = "images/" + ("train/" if train else "test/")
    wandb_img_str = "train_" if train else "test_"
    wandb_metrics = dict()
    test_data = (data['image'], data['output'], data['label'])
    grid = image_grid(5, *test_data, False, False)
    summary_writer.image(img_str + "sample", grid, epochs)
    wandb_metrics[wandb_img_str + "sample"] = wandb.Image(grid)
    grid = image_grid(5, *test_data, False, True)
    summary_writer.image(img_str + "worst", grid, epochs)
    wandb_metrics[wandb_img_str + "worst"] = wandb.Image(grid)
    if 'generated' in data.keys():
        test_data = (data['image'], data['output'], data['generated_label'])
        grid = image_grid(5, *test_data, generated=data['generated'])
        summary_writer.image(img_str + "generated", grid, epochs)
        wandb_metrics[wandb_img_str + "generated"] = wandb.Image(grid)
    if 'avg_probs' in data.keys():
        hist = histogram(data['avg_probs'])
        summary_writer.image(img_str + "avg_probs", hist, epochs)
        wandb_metrics[wandb_img_str + "avg_probs"] = wandb.Image(hist)
    if 'latents' in data.keys() and 'centers' in data.keys() and \
                data['latents'].shape[-1] == 2 and data['centers'].shape[-1] == 2:
        scatter = scatter_plot(data['latents'], data['centers'])
        summary_writer.image(img_str + "embeddings", scatter, epochs)
        wandb_metrics[wandb_img_str + "embeddings"] = wandb.Image(scatter)
    wandb.log(wandb_metrics)