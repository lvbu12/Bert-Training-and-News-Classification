# Bert-Training-and-News-Classification
Pretraining BERT and apply it to news text classification.  
实验使用Facebook实现的一个Transformer中的Encoder来完成Bert模型，链接：[fairseq](https://github.com/pytorch/fairseq)。

## Bert Training
论文地址：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Generate Data
* run `python gen_bert_index_data.py chars_vocab_path raw_text_data index_num_data`, for example, `python gen_bert_index_data.py corpus/chars.lst data/train.txt idx_data/train.txt`.

### Train Model
* run `python bert_train.py`(use default bert configuration json file path *Configs/bert.json*) or `python bert_train.py bert_config_json_file_path`


