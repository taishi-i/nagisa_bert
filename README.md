# nagisa_bert

[![Python package](https://github.com/taishi-i/nagisa_bert/actions/workflows/python-package.yml/badge.svg)](https://github.com/taishi-i/nagisa_bert/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/nagisa_bert.svg)](https://badge.fury.io/py/nagisa_bert)

This library provides a tokenizer to use [a Japanese BERT model](https://huggingface.co/taishi-i/nagisa_bert) for [nagisa](https://github.com/taishi-i/nagisa).
The model is available in [Transformers](https://github.com/huggingface/transformers) 🤗.

## Install

Python 3.7+ on Linux or macOS is required.
You can install *nagisa_bert* by using the *pip* command.


```bash
$ pip install nagisa_bert
```

## Usage

This model is available in Transformer's pipeline method.

```python
>>> from transformers import pipeline
>>> from nagisa_bert import NagisaBertTokenizer

>>> text = "nagisaで[MASK]できるモデルです"
>>> tokenizer = NagisaBertTokenizer.from_pretrained("taishi-i/nagisa_bert")
>>> fill_mask = pipeline("fill-mask", model='taishi-i/nagisa_bert', tokenizer=tokenizer)
>>> print(fill_mask(text))
[{'score': 0.1385931372642517,
  'sequence': 'nagisa で 使用 できる モデル です',
  'token': 8092,
  'token_str': '使 用'},
 {'score': 0.11947669088840485,
  'sequence': 'nagisa で 利用 できる モデル です',
  'token': 8252,
  'token_str': '利 用'},
 {'score': 0.04910655692219734,
  'sequence': 'nagisa で 作成 できる モデル です',
  'token': 9559,
  'token_str': '作 成'},
 {'score': 0.03792576864361763,
  'sequence': 'nagisa で 購入 できる モデル です',
  'token': 9430,
  'token_str': '購 入'},
 {'score': 0.026893319562077522,
  'sequence': 'nagisa で 入手 できる モデル です',
  'token': 11273,
  'token_str': '入 手'}]
```

Tokenization and vectorization.

```python
>>> from transformers import BertModel
>>> from nagisa_bert import NagisaBertTokenizer

>>> text = "nagisaで[MASK]できるモデルです"
>>> tokenizer = NagisaBertTokenizer.from_pretrained("taishi-i/nagisa_bert")
>>> tokens = tokenizer.tokenize(text)
>>> print(tokens)
['na', '##g', '##is', '##a', 'で', '[MASK]', 'できる', 'モデル', 'です']

>>> model = BertModel.from_pretrained("taishi-i/nagisa_bert")
>>> h = model(**tokenizer(text, return_tensors="pt")).last_hidden_state
>>> print(h)
tensor([[[-0.2912, -0.6818, -0.4097,  ...,  0.0262, -0.3845,  0.5816],
         [ 0.2504,  0.2143,  0.5809,  ..., -0.5428,  1.1805,  1.8701],
         [ 0.1890, -0.5816, -0.5469,  ..., -1.2081, -0.2341,  1.0215],
         ...,
         [-0.4360, -0.2546, -0.2824,  ...,  0.7420, -0.2904,  0.3070],
         [-0.6598, -0.7607,  0.0034,  ...,  0.2982,  0.5126,  1.1403],
         [-0.2505, -0.6574, -0.0523,  ...,  0.9082,  0.5851,  1.2625]]],
       grad_fn=<NativeLayerNormBackward0>)
```
