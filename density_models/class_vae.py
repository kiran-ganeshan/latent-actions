from absl import flags
import wandb, sys
from models import ClassVAELearner, train

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('num_enc_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('num_dec_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_float('enc_hidden_size', 0.5, 'Encoder hidden layer size (proportion of input).')
flags.DEFINE_float('dec_hidden_size', 0.5, 'Decoder hidden layer size (proportion of output).')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
 
wandb.init()
FLAGS(sys.argv)
coder = ClassVAELearner(num_enc_layers=FLAGS.num_enc_layers,
                        num_dec_layers=FLAGS.num_dec_layers,
                        enc_hidden_size=FLAGS.enc_hidden_size,
                        dec_hidden_size=FLAGS.dec_hidden_size,
                        learning_rate=FLAGS.learning_rate,
                        beta1=FLAGS.beta2,
                        beta2=FLAGS.beta1,
                        beta=FLAGS.beta,
                        latent_dim=FLAGS.latent_dim)
train(coder, 
      FLAGS.epochs, 
      FLAGS.batch_size, 
      FLAGS.test_interval, 
      FLAGS.seed, 
      FLAGS.save_dir)
wandb.finish()