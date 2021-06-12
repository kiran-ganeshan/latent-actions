from absl import flags
import wandb, sys
from models import MADELearner, train

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 784, 'Dimension of latent space.')
flags.DEFINE_integer('num_values', 2, 'Number of categorical latent values.')
flags.DEFINE_integer('num_layers', 2, 'Number of layers in Masked MLP.')
flags.DEFINE_integer('hidden_size', 2, 'Size of hidden layers (integral proportion of input/output).')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
 
wandb.init()
FLAGS(sys.argv)
coder = MADELearner(learning_rate=FLAGS.learning_rate,
                    beta1=FLAGS.beta2,
                    beta2=FLAGS.beta1,
                    latent_dim=FLAGS.latent_dim,
                    num_values=FLAGS.num_values)
train(coder, 
      FLAGS.epochs, 
      FLAGS.batch_size, 
      FLAGS.test_interval, 
      FLAGS.seed, 
      FLAGS.save_dir)
wandb.finish()