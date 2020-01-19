from .feed_forward import FeedForward
from transformer_contrib.backend import utils

utils.get_custom_objects().update(
    {
        'FeedForward' : FeedForward,
        }
    )
