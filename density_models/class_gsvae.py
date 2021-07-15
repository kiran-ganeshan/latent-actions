from absl import flags
import wandb, sys
from models import ClassGSVAELearner
from utils import get_mnist_datasets, train

FLAGS = flags.FLAGS
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('num_values', 10, 'Number of values of categorical latents.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('temp_interval', 2, 'Temperature update interval (epochs).')
flags.DEFINE_integer('num_enc_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('num_dec_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_float('enc_hidden_size', 0.5, 'Encoder hidden layer size (proportion of input).')
flags.DEFINE_float('dec_hidden_size', 0.5, 'Decoder hidden layer size (proportion of output).')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('lr_decay', 5e-4, 'Learning rate exponential decay (per epoch).')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
flags.DEFINE_float('max_temp', 0.5, 'Maximum temperature.')
flags.DEFINE_float('temp_rate', 0.1, 'Rate of temperature decrease (per epoch).')
 
wandb.init()
FLAGS(sys.argv)
coder = ClassGSVAELearner(input_size=784,
                          input_shape=(28, 28, 1), 
                          cond_size=10,
                          output_size=784,
                          num_enc_layers=FLAGS.num_enc_layers,
                          num_dec_layers=FLAGS.num_dec_layers,
                          enc_hidden_size=FLAGS.enc_hidden_size,
                          dec_hidden_size=FLAGS.dec_hidden_size,
                          learning_rate=FLAGS.learning_rate,
                          lr_decay=FLAGS.lr_decay,
                          beta1=FLAGS.beta2,
                          beta2=FLAGS.beta1,
                          beta=FLAGS.beta,
                          max_temp=FLAGS.max_temp,
                          temp_rate=FLAGS.temp_rate,
                          temp_interval=FLAGS.temp_interval,
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