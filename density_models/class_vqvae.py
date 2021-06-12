from absl import flags
import wandb, sys
from models import VQVAELearner, train

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('test_interval', 5, 'Testing interval (epochs).')
flags.DEFINE_integer('latent_dim', 10, 'Dimension of latent space.')
flags.DEFINE_integer('num_values', 10, 'Number of values for categorical latent variables.')
flags.DEFINE_integer('embedding_dim', 2, 'Dimension of embedding space.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('seed', 0, 'Psuedorandom generator key.')
flags.DEFINE_integer('batch_size', 60, 'Batch size.')
flags.DEFINE_integer('made_epochs', 10, 'Number of epochs for MADE.')
flags.DEFINE_integer('made_batch', 60, 'Batch size for MADE.')
flags.DEFINE_integer('made_num_layers', 2, 'Number of layers in MADE.')
flags.DEFINE_integer('made_hidden_size', 2, 'Size of MADE hidden layers as a multiple of latent_dim*num_values.')
flags.DEFINE_integer('num_enc_layers', 2, 'Number of layers in encoder.')
flags.DEFINE_integer('num_dec_layers', 2, 'Number of layers in decoder.')
flags.DEFINE_float('enc_hidden_size', 0.5, 'Encoder hidden layer size (proportion of input).')
flags.DEFINE_float('dec_hidden_size', 0.5, 'Decoder hidden layer size (proportion of output).')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate.')
flags.DEFINE_float('beta1', 0.5, 'First Adam parameter.')
flags.DEFINE_float('beta2', 0.9, 'Second Adam parameter.')
flags.DEFINE_float('made_learning_rate', 1e-5, 'MADE Learning rate.')
flags.DEFINE_float('made_beta1', 0.9, 'First Adam param for MADE.')
flags.DEFINE_float('made_beta2', 0.5, 'Second Adam param for MADE.')
flags.DEFINE_float('beta', 0.5, 'KL loss term scaling factor.')
flags.DEFINE_float('commitment_cost', 0.25, 'VQVAE Quantizer commitment cost.')
flags.DEFINE_float('made_coeff', 0.2, 'Importance of MADE in aggregate loss.')
 
wandb.init()
FLAGS(sys.argv)
assert FLAGS.made_num_layers >= 1
coder = VQVAELearner(num_enc_layers=FLAGS.num_enc_layers,
                     num_dec_layers=FLAGS.num_dec_layers,
                     enc_hidden_size=FLAGS.enc_hidden_size,
                     dec_hidden_size=FLAGS.dec_hidden_size,
                     latent_dim=FLAGS.latent_dim,
                     num_values=FLAGS.num_values,
                     embedding_dim=FLAGS.embedding_dim,
                     num_classes=FLAGS.num_classes,
                     learning_rate=FLAGS.learning_rate,
                     beta1=FLAGS.beta2,
                     beta2=FLAGS.beta1,
                     made_learning_rate=FLAGS.made_learning_rate,
                     made_beta1=FLAGS.made_beta1,
                     made_beta2=FLAGS.made_beta2,
                     beta=FLAGS.beta,
                     commitment_cost=FLAGS.commitment_cost,
                     made_epochs=FLAGS.made_epochs,
                     made_batch=FLAGS.made_batch,
                     made_num_layers=FLAGS.made_num_layers,
                     made_hidden_size=FLAGS.made_hidden_size,
                     ema_vq=False)
train(coder, 
      FLAGS.epochs, 
      FLAGS.batch_size, 
      FLAGS.test_interval, 
      FLAGS.seed, 
      FLAGS.save_dir, 
      FLAGS.made_coeff)
wandb.finish()