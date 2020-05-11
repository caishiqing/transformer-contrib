# Transformer系列模型keras集成

本项目集成了keras官方Transformer系列模型包，其中大部分来自于[CyberZHG](https://github.com/CyberZHG)，由于每个模型和网络层都是独立的安装包，安装比较繁琐，所以本项目对这些包做了集成封装，所有的自定义类型自动注册，可以方便地序列化与加载。如果要使用 `tf.keras`，加入配置

```python
os.environ['TF_KERAS'] = '1'
```

如果使用 tf2.x ，则默认开启 TF_KERAS，不需要手动配置。

使用 tf1.x 建议 使用`tf.keras`或者 tf=1.12.0 + keras=2.2.4，其他版本组合没有详细测试。

## 更新说明

**keras_bert: ** 修改了Tokenizer的rematch方法，将复杂度从O(N^2)降低为O(N).

**keras_gpt_2:**  增加了pad_id（默认为None）参数和return_logits（默认为False）参数，pad_id指定需要mask的输入token，如果为None则无mask；return_logits指示最后一层输出logits还是概率分布。

**keras_layers:**  ScaledDotProductAttention层修改attention mask分母添加epsilon的方式，原方式会导致注意力的误差累加，对gpt2的因果注意力造成影响。



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

Transformer模型定义在transformer_contrib.keras_transformer中，模型与 **Attention is all your need** 论文中一致，具体调用实例如下

```python
from transformer_contrib.keras_transformer import get_model

# Build the model
# model是综合模型，encoder、decoder分别为对应的编码器解码器子模块
model, encoder, decoder = get_model(
    token_num=10000,
    embed_dim=30,
    encoder_num=3,
    decoder_num=2,
    head_num=3,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
)
model.summary()

```



## BERT

BERT模型的定义与预训练模型加载以及分词器接口在transformer_contrib.keras_bert中，分词器调用实例：

```python
from transformer_contrib.keras_bert import Tokenizer

# 从指定词典文件加载
tokenizer = Tokenizer.from_file('../vocab.txt')
# 常规分词
tokens = tokenizer._tokenize('姚明身高221cm')
# >> ['姚', '明', '身', '高', '221', '##cm']

# 完整分词
tokens = tokenizer.tokenize('姚明身高221cm')
# >> ['[CLS]', '姚', '明', '身', '高', '221', '##cm', '[SEP]']

# 句子对分词
tokens = tokenizer.tokenize('姚明多高？', '姚明身高226cm')
# >> ['[CLS]', '姚', '明', '多', '高', '？', '[SEP]', '姚', '明', '身', '高', '226', '##cm', '[SEP]']

# 也可以直接编码，编码序列与分词结果对应，返回 sequence 与 segment 两个对象
seq, seg = tokenizer.encode('姚明多高？', '姚明身高226cm')
# >> seq: [101, 2001, 3209, 1914, 7770, 8043, 102, 2001, 3209, 6716, 7770, 10436, 8341, 102]
# >> seg: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# 编码的同时可以把最大长度填充也给做了
seq, seg = tokenizer.encode('姚明多高？', '姚明身高226cm', max_len=18)
# >> seq: [101, 2001, 3209, 1914, 7770, 8043, 102, 2001, 3209, 6716, 7770, 10436, 8341, 102, 0, 0, 0, 0]
# >> seg: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# 如果句子长度超过最大长度需要做截断，会动态地截断两个句子中较长的那个

# 有些任务需要标注句子中的子片段
text = '姚明身高221cm，被称为小巨人。'
s, e = 4, 8  # 221cm
tokens = tokenizer._tokenize(text)  # 不要加[CLS]和[SEP]
# >> ['姚', '明', '身', '高', '221', '##cm', '，', '被', '称', '为', '小', '巨', '人', '。']
intervals = tokenizer.rematch(text, tokens)  # 每一个token对应原字符串的（起始索引，结束索引）
token_bound = tokenizer.transform_bound(intervals, start=s, end=e)
# >> token_bound = (4, 5)
# 输出“221cm”这个子串对应的 token 子序列的首尾位置，注意是位置不是片段索引，尾部位置=尾部片段索引-1：
# tokens[4: 5] = ['221'], tokens[4: 6] = ['221', 'cm']
```



模型训练与使用（以文本分类为例）：

```python
# 如果要使用 tf.keras（tf2.x不需要配置）,需要在导入transformer_contrib包之前配置：
# import os
# os.['TF_KERAS'] = '1'
from transformer_contrib.keras_bert import load_bert_from_ckpt, Tokenizer
from transformer.backend import keras
import tensorflow as tf

checkpoint_path = '../uncased_L-12_H-768_A-12'

bert = load_bert_from_ckpt(
    checkpoint_path,
    training=False,  # 去掉顶层网络
    trainable=True,  # 参数可训练
)
h = keras.layers.Lambda(lambda x:x[:,0,:])(bert.output)
y = keras.layers.Dense(10, activation='softmax')(h)
model = keras.Model(bert.inputs, y)
model.summary()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')

tokenizer = Tokenizer.from_file('../uncased_L-12_H-768_A-12/vocab.txt')

def get_dataset(data, batch_size=16, max_len=128):
    # data: list of (text, label)
    import random
    random.shuffle(data)
    seqs, segs, labels = [], [], []
    for text, label in data:
        seq, seg = tokenizer.encode(text, max_len=max_len)
        seqs.append(seq)
        segs.append(seg)
        labels.append(label)
    
    dataset = tf.data.Dataset.from_tensoror_slices(
        ((seqs, segs), labels)).batch(batch_size, drop_remainder=True)
    return dataset

# 假设已经加载了训练集和验证集
train_dataset = get_dataset(train_data)
valid_dataset = get_dataset(valid_data)
model.fit(
    x=train_dataset,
    validation_data=valid_dataset,
    epochs=10,
)
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

