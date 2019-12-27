from .backend import keras, utils
import tensorflow as tf

__all__ = [
    'Concat', 'Stack', 'GatherLayer',
]


class Concat(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.support_masking = True
        
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            return tf.concat(inputs, axis=-2)
        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if isinstance(mask, list):
            return tf.concat(mask, axis=-1)
        return mask

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            B, len1, dim = input_shape[0]
            B, len2, dim = input_shape[1]
            return B, len1 + len2, dim
        return input_shape
            
    
class Stack(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.support_masking = True

    def call(self, inputs, **kwargs):
        return tf.stack(inputs, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.stack(mask, axis=1)

    def compute_output_shape(self, input_shape):
        batch, length, dim = input_shape[0]
        return batch, len(input_shape), length, dim


class GatherLayer(keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super().__init(**kwargs)
        self.n = n
        self.support_masking = True
        
    def call(self, inputs, **kwargs):
        if inputs is None:
            return None
        return inputs[:, self.n]
        
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if inputs is None:
            return None
        return mask[:, self.n]
        
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3]


utils.get_custom_objects().update(
    {
        'Concat': Concat,
        'Stack': Stack,
        'GatherLayer': GatherLayer,
    }
)