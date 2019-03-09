# Bert-Training-and-News-Classification
Pretraining BERT and apply it to character level chinese news text classification.  
实验使用Facebook实现的一个Transformer中的Encoder来完成Bert模型，链接：[fairseq](https://github.com/pytorch/fairseq)。

## Bert Training
论文地址：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Generate Data
* run `python gen_bert_index_data.py chars_vocab_path raw_text_data index_num_data`.  For example, `python gen_bert_index_data.py corpus/chars.lst data/train.txt idx_data/train.txt`.

### Train Model
* change the train data path or valid data path or other settings in *Configs/bert.json*, or use the default settings.
* run `python bert_train.py`(use default bert configuration json file path *Configs/bert.json*) or `python bert_train.py bert_config_json_file_path`

### Test Model
* change the test data path or prediction output path in *Configs/bert.json*, or use the default settings.
* run `python bert_test.py`, compare the mask prediction and is_next_sent label with raw text stored in data directory, and compute the accuray of prediction.

## Chinese News Classification
### Train Model
* change the train data path or valid data path or other settings in *Configs/para_cls.json*, or use the default settings.
* run `python Chinese_news_cls_train.py`(use default configuration json file path *Configs/para_cls.json*) or `python Chinese_news_cls_train.py custom_config_json_file_path`

### Test Model
* change the test data path or other settings in *Configs/para_cls.json*, or use the default settings.
* run `python Chinese_news_cls_test.py`(use default configuration json file path *Configs/para_cls.json*) or `python Chinese_news_cls_test.py custom_config_json_file_path`

### Report F1 Score
* Run function `gen_csv_report` in report.py to get the report csv which contains the confusion matrix.
* Run function `compute_macro_F1` or 'compute_micro_F1' to get the macro F1 score or micro F1 score from the confusion matrix.

