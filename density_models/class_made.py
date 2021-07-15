from absl import flags
import wandb, sys
from density_models import CondMADELearner
from utils import get_mnist_datasets, train

FLAGS = flags.FLAGS
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 784, 'Dimension of latent space.')
flags.DEFINE_integer('num_values', 2, 'Number of categorical latent values.')
flags.DEFINE_integer('num_layers', 2, 'Number of layers in Masked MLP.')
flags.DEFINE_integer('hidden_size', 2, 'Size of hidden layers (integral proportion of input/output).')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('lr_decay', 5e-4, 'Learning rate exponential decay (per epoch).')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
 
wandb.init()
FLAGS(sys.argv)
coder = CondMADELearner(input_size=784,
                        input_shape=(28, 28, 1), 
                        cond_size=10,
                        learning_rate=FLAGS.learning_rate,
                        lr_decay=FLAGS.lr_decay,
                        beta1=FLAGS.beta2,
                        beta2=FLAGS.beta1,
                        latent_dim=FLAGS.latent_dim,
                        num_values=FLAGS.num_values)
train_ds, test_ds = get_mnist_datasets()
train(coder, 
      FLAGS.epochs, 
      FLAGS.batch_size, 
      FLAGS.test_interval, 
      (train_ds['image'],
       train_ds['label']),
      (test_ds['image'],
       test_ds['label']),
      FLAGS.seed)
wandb.finish()