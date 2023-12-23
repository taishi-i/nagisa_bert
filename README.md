# nagisa_bert

[![Python package](https://github.com/taishi-i/nagisa_bert/actions/workflows/python-package.yml/badge.svg)](https://github.com/taishi-i/nagisa_bert/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/nagisa_bert.svg)](https://badge.fury.io/py/nagisa_bert)

This library provides a tokenizer to use [a Japanese BERT model](https://huggingface.co/taishi-i/nagisa_bert) for [nagisa](https://github.com/taishi-i/nagisa).
The model is available in [Transformers](https://github.com/huggingface/transformers) 🤗.

You can try fill-mask using nagisa_bert at [Hugging Face Space](https://huggingface.co/spaces/taishi-i/nagisa_bert-fill_mask).


## Install

Python 3.7+ on Linux or macOS is required.
You can install *nagisa_bert* by using the *pip* command.


```bash
$ pip install nagisa_bert
```

## Usage

This model is available in Transformer's pipeline method.

```python
from transformers import pipeline
from nagisa_bert import NagisaBertTokenizer

text = "nagisaで[MASK]できるモデルです"
tokenizer = NagisaBertTokenizer.from_pretrained("taishi-i/nagisa_bert")
fill_mask = pipeline("fill-mask", model='taishi-i/nagisa_bert', tokenizer=tokenizer)
print(fill_mask(text))
```

```python
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
from transformers import BertModel
from nagisa_bert import NagisaBertTokenizer

text = "nagisaで[MASK]できるモデルです"
tokenizer = NagisaBertTokenizer.from_pretrained("taishi-i/nagisa_bert")
tokens = tokenizer.tokenize(text)
print(tokens)
# ['na', '##g', '##is', '##a', 'で', '[MASK]', 'できる', 'モデル', 'です']

model = BertModel.from_pretrained("taishi-i/nagisa_bert")
h = model(**tokenizer(text, return_tensors="pt")).last_hidden_state
print(h)
```

```python
tensor([[[-0.2912, -0.6818, -0.4097,  ...,  0.0262, -0.3845,  0.5816],
         [ 0.2504,  0.2143,  0.5809,  ..., -0.5428,  1.1805,  1.8701],
         [ 0.1890, -0.5816, -0.5469,  ..., -1.2081, -0.2341,  1.0215],
         ...,
         [-0.4360, -0.2546, -0.2824,  ...,  0.7420, -0.2904,  0.3070],
         [-0.6598, -0.7607,  0.0034,  ...,  0.2982,  0.5126,  1.1403],
         [-0.2505, -0.6574, -0.0523,  ...,  0.9082,  0.5851,  1.2625]]],
       grad_fn=<NativeLayerNormBackward0>)
```

## Tutorial

You can find here a list of the notebooks on Japanese NLP using pre-trained models and transformers.


| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Fill-mask](https://github.com/taishi-i/nagisa_bert/blob/develop/notebooks/fill_mask-japanese_bert_models.ipynb)  | How to use the pipeline function in transformers to fill in Japanese text. |[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taishi-i/nagisa_bert/blob/develop/notebooks/fill_mask-japanese_bert_models.ipynb)|
| [Feature-extraction](https://github.com/taishi-i/nagisa_bert/blob/develop/notebooks/feature_extraction-japanese_bert_models.ipynb)  | How to use the pipeline function in transformers to extract features from Japanese text. |[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taishi-i/nagisa_bert/blob/develop/notebooks/feature_extraction-japanese_bert_models.ipynb)|
| [Embedding visualization](https://github.com/taishi-i/nagisa_bert/blob/develop/notebooks/embedding_visualization-japanese_bert_models.ipynb)  | Show how to visualize embeddings from Japanese pre-trained models. |[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taishi-i/nagisa_bert/blob/develop/notebooks/embedding_visualization_japanese_bert_models.ipynb)|
| [How to fine-tune a model on text classification](https://github.com/taishi-i/nagisa_bert/blob/develop/notebooks/text_classification-amazon_reviews_ja.ipynb)  | Show how to fine-tune a pretrained model on a Japanese text classification task. |[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taishi-i/nagisa_bert/blob/develop/notebooks/text_classification-amazon_reviews_ja.ipynb)|
| [How to fine-tune a model on text classification with csv files](https://github.com/taishi-i/nagisa_bert/blob/develop/notebooks/text_classification-csv_files.ipynb)  | Show how to preprocess the data and fine-tune a pretrained model on a Japanese text classification task. |[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taishi-i/nagisa_bert/blob/develop/notebooks/text_classification-csv_files.ipynb)|


## Model description

### Architecture

The model architecture is the same as [the BERT bert-base-uncased architecture](https://huggingface.co/bert-base-uncased) (12 layers, 768 dimensions of hidden states, and 12 attention heads).

### Training Data

The models is trained on the Japanese version of Wikipedia. The training corpus is generated from the Wikipedia Cirrussearch dump file as of August 8, 2022 with [make_corpus_wiki.py](https://github.com/cl-tohoku/bert-japanese/blob/main/make_corpus_wiki.py) and [create_pretraining_data.py](https://github.com/cl-tohoku/bert-japanese/blob/main/create_pretraining_data.py).

### Training

The model is trained with the default parameters of [transformers.BertConfig](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig).
Due to GPU memory limitations, the batch size is set to small; 16 instances per batch, and 2M training steps.
