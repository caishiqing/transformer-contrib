from .inputs import get_inputs
from .embedding import get_embedding, TokenEmbedding, EmbeddingSimilarity
from .masked import Masked
from .extract import Extract
from .pooling import MaskedGlobalMaxPool1D
from .conv import MaskedConv1D
from transformer_contrib.backend import keras


keras.utils.get_custom_objects().update(
    {
        'TokenEmbedding': TokenEmbedding,
        'EmbeddingSimilarity': EmbeddingSimilarity,
        'Masked': Masked,
        'Extract': Extract,
        'MaskedGlobalMaxPool1D': MaskedGlobalMaxPool1D,
        'MaskedConv1D': MaskedConv1D
        }
)
