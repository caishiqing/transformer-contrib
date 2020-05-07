import os
import tensorflow as tf

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets', 'engine',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers', 'TF_KERAS', 'EAGER_MODE'
]

TF_KERAS = False
EAGER_MODE = False

if tf.__version__.split('.')[0] == '2':
    os.environ['TF_KERAS'] = '1'

if os.environ.get('TF_KERAS', '0') != '0':
    from tensorflow.python import keras
    TF_KERAS = True
    [tf.enable_eager_execution() for _ in range(1)
     if not tf.executing_eagerly() and os.environ.get('TF_EAGER', '0') != '0']
    EAGER_MODE = tf.executing_eagerly()
else:
    import keras

utils = keras.utils
activations = keras.activations
applications = keras.applications
backend = keras.backend
datasets = keras.datasets
engine = keras.engine
layers = keras.layers
preprocessing = keras.preprocessing
wrappers = keras.wrappers
callbacks = keras.callbacks
constraints = keras.constraints
initializers = keras.initializers
metrics = keras.metrics
models = keras.models
losses = keras.losses
optimizers = keras.optimizers
regularizers = keras.regularizers
