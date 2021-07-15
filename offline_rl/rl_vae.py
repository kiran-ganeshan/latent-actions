from absl import flags
import wandb, sys, gym
import numpy as np
from rl_models import RLVLearner
from utils import get_rl_datasets, train

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'halfcheetah-expert-v0', 'D4RL environment name.')
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('num_enc_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('num_dec_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_integer('num_evals', 100, 'Number of Q-function evaluations each learning step.')
flags.DEFINE_float('enc_hidden_size', 0.5, 'Encoder hidden layer size (proportion of input).')
flags.DEFINE_float('dec_hidden_size', 0.5, 'Decoder hidden layer size (proportion of output).')
flags.DEFINE_float('gamma', 0.9, 'Stability factor in Bellman optimality operator.')
flags.DEFINE_float('learning_rate', 1e-7, 'Learning rate.')
flags.DEFINE_float('lr_decay', 1, 'Learning rate exponential decay (over all epochs).')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
 
wandb.init()
FLAGS(sys.argv)
obs_size = np.prod(gym.make(FLAGS.env).observation_space.shape)
act_shape = gym.make(FLAGS.env).action_space.shape
coder = RLVLearner(input_size=np.prod(act_shape),
                   input_shape=act_shape,
                   cond_size=obs_size,
                   num_enc_layers=FLAGS.num_enc_layers,
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
                   num_evals=FLAGS.num_evals)
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