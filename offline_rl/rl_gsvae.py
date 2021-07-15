from absl import flags
import wandb, sys
from rl_models import RLGSVLearner
from utils import get_rl_datasets, train

FLAGS = flags.FLAGS
flags.DEFINE_integer('env', 'Cartpole-v0', 'D4RL environment name.')
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('num_enc_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('num_dec_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_float('enc_hidden_size', 0.5, 'Encoder hidden layer size (proportion of input).')
flags.DEFINE_float('dec_hidden_size', 0.5, 'Decoder hidden layer size (proportion of output).')
flags.DEFINE_float('gamma', 0.99, 'Stability factor in Bellman optimality operator.')
flags.DEFINE_float('num_eval', 10, 'Number of Q-function evaluations each learning step.')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('lr_decay', 5e-4, 'Learning rate exponential decay (per epoch).')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
 
wandb.init()
FLAGS(sys.argv)
coder = RLGSVLearner(num_enc_layers=FLAGS.num_enc_layers,
                     num_dec_layers=FLAGS.num_dec_layers,
                     enc_hidden_size=FLAGS.enc_hidden_size,
                     dec_hidden_size=FLAGS.dec_hidden_size,
                     learning_rate=FLAGS.learning_rate,
                     lr_decay=FLAGS.lr_decay,
                     beta1=FLAGS.beta2,
                     beta2=FLAGS.beta1,
                     beta=FLAGS.beta,
                     latent_dim=FLAGS.latent_dim,
                     gamma=FLAGS.gamma,
                     num_eval=FLAGS.num_eval)
train_ds, test_ds = get_rl_datasets(FLAGS.env)
train(coder, 
      FLAGS.epochs, 
      FLAGS.batch_size, 
      FLAGS.test_interval, 
      (train_ds['actions'],
       train_ds['observations'],
       train_ds['rewards'],
       train_ds['next_observations']),
      (test_ds['actions'],
       test_ds['observations'],
       test_ds['rewards'],
       test_ds['next_observations']),
      FLAGS.seed)
wandb.finish()