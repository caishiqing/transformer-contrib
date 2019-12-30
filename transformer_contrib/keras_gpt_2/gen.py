import numpy as np
import tensorflow as tf

__all__ = [
    'generate', 'top_k_logits', 'top_p_logits',
]

def generate(model,
             bpe,
             texts,
             length=100,
             top_k=1,
             temperature=1.0):
    """Generate text after the given contexts.

    :param model: The trained model.
    :param bpe: Byte pair encoding object.
    :param texts: A list of texts.
    :param length: The length of following texts to be generated.
    :param top_k: Choose the next token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :return: A list of generated texts.
    """
    batch_size = len(texts)
    encodes = [bpe.encode(text) for text in texts]
    text_lens = [len(encode) for encode in encodes]
    max_len = max(text_lens)
    input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes]
    for shift in range(length):
        output_data = model.predict(np.array(input_data))
        for index in range(batch_size):
            probs = [(prob, i) for i, prob in enumerate(output_data[index, text_lens[index] + shift - 1])]
            probs.sort(reverse=True)
            probs = probs[:top_k]
            indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
            probs = np.array(probs) / temperature
            probs = probs - np.max(probs)
            probs = np.exp(probs)
            probs = probs / np.sum(probs)
            next_token = np.random.choice(indices, p=probs)
            input_data[index].append(0)
            input_data[index][text_lens[index] + shift] = next_token
    outputs = [bpe.decode(input_data[index][:text_lens[index] + length]) for index in range(batch_size)]
    return outputs


def top_k_logits(logits, k):
    if k == 0:
    # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(tf.equal(k, 0),
                    lambda: logits,
                    lambda: _top_k())


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.contrib.framework.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
                logits < min_values,
                tf.ones_like(logits) * -1e10,
                logits,
            )