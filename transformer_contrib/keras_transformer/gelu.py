import math
import tensorflow as tf

def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    #return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))
