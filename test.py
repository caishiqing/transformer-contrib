

def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]

def unpack_singleton(x):
    """Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x

def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    input_tensors = to_list(input_tensors)
    masks = []
    for x in input_tensors:
        if hasattr(x, '_keras_history'):
            inbound_layer, node_index, tensor_index = x._keras_history
            node = inbound_layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    return unpack_singleton(masks)


if __name__ == '__main__':
    from transformer_contrib.keras_gpt_2 import get_gpt2
    import keras
    import keras.backend as K
    import tensorflow as tf

    x = tf.constant([[0,0,1,2,3]], dtype='float32')
    gpt = get_gpt2(100,10,pad_id=0)
    y = gpt(x)
