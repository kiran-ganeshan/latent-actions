import jax
from jax import numpy as jnp
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data, color
import io

matplotlib.use('agg')

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


def one_hot(label):
    return (label[..., None] == jnp.arange(10)[None]).astype(jnp.float32)


def write_summary(summary_writer, metrics, rng, epochs, train=False, images=False):
    scalar_keys = metrics.keys() - ['image', 'output', 'label', 'generated']
    for key in scalar_keys:
        metric_str = key + "/" + ("train" if train else "test")
        summary_writer.scalar(metric_str, metrics[key], epochs)
    if train:
        summary_str = str()
        for key in scalar_keys:
            summary_str += "{0}: {1}, ".format(key, metrics[key])
        print(summary_str[:-2])
    if images:
        print(metrics['image'][0, ...].min(), metrics['image'][0, ...].max())
        print(metrics['output'][0, ...].min(), metrics['output'][0, ...].max())
        img_str = "images/" + ("train/" if train else "test/")
        data = (metrics['image'], metrics['output'], metrics['label'])
        grid = image_grid(5, *data, False, False)
        summary_writer.image(img_str + "sample", grid, epochs)
        grid = image_grid(5, *data, False, True)
        summary_writer.image(img_str + "worst", grid, epochs)
    if images and 'generated' in metrics.keys():
        grid = image_grid(5, *data, generated=metrics['generated'])
        summary_writer.image(img_str + "generated", grid, epochs)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    width, height = figure.get_size_inches() * figure.get_dpi()
    #figure.tight_layout()
    figure.canvas.draw()
    im = np.fromstring(figure.canvas.tostring_rgb(), dtype='uint8')
    im = im.reshape(int(height), int(width), 3)
    plt.close(figure)
    return color.rgb2gray(im)

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
    figure = plt.figure(figsize=(4 * n, 4 * n))
    use_worst = worst and generated is None
    for (i, index) in enumerate(worst_image_it if use_worst else range(n * n)):
        # Start next subplot.
        if feed_forward and generated is None:
            title = "Predicted: {0}\nActual: {1}".format(output[index], label[index])
        elif generated is None:
            title = "Label: {}".format(label[index])
        else:
            title = "Generated"
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