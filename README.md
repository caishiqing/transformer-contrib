# Transformer系列模型keras集成

本项目集成了keras官方Transformer系列模型包，其中大部分来自于[CyberZHG](https://github.com/CyberZHG)，由于每个模型和网络层都是独立的安装包，安装比较繁琐，所以本项目对这些包做了集成封装，所有的自定义类型自动添加到keras类型中，可以方便地序列化与加载。如果要使用 `tensorflow.python.keras`，加入配置

```python
os.environ['TF_KERAS'] = '1'
```



## 基础网络层

基础网络层定义在transformer_contrib.keras_layers中，包括keras_pos_embd，keras_layer_normalization，keras_embed_sim，keras_multi_head，keras_self_attention等，以keras_multi_head为例，调用方式为：

```python
from transformer_contrib.keras_layers import keras_multi_head
from keras.layers import Input

x = Input(shape=(100, 256))
# 函数式API
att = MultiHeadAttention(head_num=8, activation='relu')(x)
```



## Transformer

Transformer模型定义在transformer_contrib.keras_transformer中，模型与**Attention is all your need**论文中一致，具体调用实例如下

```python
from transformer_contrib.keras_transformer import get_model

# Build the model
model = get_model(
    token_num=10000,
    embed_dim=30,
    encoder_num=3,
    decoder_num=2,
    head_num=3,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((13, 30)),
)
model.summary()

```



## BERT

BERT模型的定义与预训练模型加载以及分词器接口在transformer_contrib.keras_bert中，分词器调用实例：

```python
from transformer_contrib.keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}
tokenizer = Tokenizer(token_dict)  
# from transformer_contrib.keras_bert import load_tokenizer
# tokenizer = load_tokenizer('xxx/chinese_L-12_H-768_A-12/tokenizer.txt')
print(tokenizer.tokenize('unaffable'))  # The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]']`
indices, segments = tokenizer.encode('unaffable')
print(indices)  # Should be `[0, 2, 3, 4, 1]`
print(segments)  # Should be `[0, 0, 0, 0, 0]`

print(tokenizer.tokenize(first='unaffable', second='钢'))
# The result should be `['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']`
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)  # Should be `[0, 2, 3, 4, 1, 5, 1, 0, 0, 0]`
print(segments)  # Should be `[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]`
```

模型训练与使用：

```python
import os
from transformer_cpntrib.keras_bert import load_bert_from_ckpt
from keras.layers import Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam

layer_num = 12
checkpoint_path = '../uncased_L-12_H-768_A-12'

bert = load_bert_from_ckpt(
    checkpoint_path,
    training=False,
    use_adapter=True,
    trainable=True,
)
h = Lambda(lambda x:x[:,0,:])(bert.output)
y = Dense(10, activation='softmax')(h)
model = Model(bert.inputs, y)
model.summary()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')
```



## Transformer-xl

Transformer-xl定义在transformer_contrib.keras_transformer_xl中，调用方法为：

```python
import keras
import numpy as np
from transformer_contrib.keras_transformer_xl import MemorySequence, build_transformer_xl


class DummySequence(keras.utils.Sequence):

    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


model = build_transformer_xl(
    units=4,
    embed_dim=4,
    hidden_dim=4,
    num_token=3,
    num_block=3,
    num_head=2,
    batch_size=3,
    memory_len=20,
    target_len=10,
)
seq = MemorySequence(
    model=model,
    sequence=DummySequence(),
    target_len=10,
)

model.predict(model, seq, verbose=True)
```

加载预训练模型：

```python
import os
from transformer_contrib.keras_transformer_xl import load_trained_model_from_checkpoint

checkpoint_path = 'foo/bar/sota/enwiki8'
model = load_trained_model_from_checkpoint(
    config_path=os.path.join(checkpoint_path, 'config.json'),
    checkpoint_path=os.path.join(checkpoint_path, 'model.ckpt')
)
model.summary()
```



## GPT2

GPT2模型定义在transformer_contrib.keras_gpt_2中， 调用示例：

```python
import os
from transformer_contrib.keras_gpt_2 import load_gpt2_from_ckpt, get_bpe_from_files, generate

ckpt_path = 'xxx/yyy/117M'

print('Load model from checkpoint...')
model = load_gpt2_from_ckpt(ckpt_path)
print('Load BPE from files...')
bpe = load_bpe_from_ckpt(ckpt_path)
print('Generate text...')
output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=1)

# If you are using the 117M model and top_k equals to 1, then the result will be:
# "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
print(output[0])
```



## XLNet

XLNet定义在transformer_contrib.keras_xlnet中，调用如下：

```python
import os
from transformer_contrib.keras_xlnet import Tokenizer, load_xlnet_from_ckpt, ATTENTION_TYPE_BI

checkpoint_path = '.../xlnet_cased_L-24_H-1024_A-16'

tokenizer = Tokenizer(os.path.join(checkpoint_path, 'spiece.model'))
model = load_xlnet_from_ckpt(
    ckpt_path=checkpoint_path,
    batch_size=16,
    memory_len=512,
    target_len=128,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)
model.summary()
```

