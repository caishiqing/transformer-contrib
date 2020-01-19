from .gelu import gelu
from .transformer import *
from transformer_contrib.backend import utils

utils.get_custom_objects().update({'gelu': gelu})
