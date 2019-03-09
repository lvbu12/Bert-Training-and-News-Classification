# Bert-Training-and-News-Classification
Pretraining BERT and apply it to news text classification.  
实验使用Facebook实现的一个Transformer中的Encoder来完成Bert模型，链接：[fairseq](https://github.com/pytorch/fairseq)。

## Bert Training
论文地址：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Generate Data
* run `python gen_bert_index_data.py your_chars_vocab_path your_raw_text_data your_index_num_data`

