# ArgueBERT
ArgueBERT is a pre-trained and fine-tuned model based on Sentence BERT by Reimers and Gurevych [[1]](https://aclanthology.org/D19-1410/) to measure the similarity of arguments.
You can download the pre-trained PyTorch models here:
* [Similarity prediction argueBERT](https://uni-duesseldorf.sciebo.de/s/bxaCDV5B9SBzFqH/download)
* [Similarity prediction argueBERT<sub>BASE</sub>](https://uni-duesseldorf.sciebo.de/s/ZTkI8umuMsd5nxv/download)
* [Argument order prediction argueBERT](https://uni-duesseldorf.sciebo.de/s/qv8IJ1pjrSBIuGE/download)
* [Argument graph edge validation argueBERT](https://uni-duesseldorf.sciebo.de/s/HquFKqgrKmzwlCd/download)
* [Baseline SBERT](https://uni-duesseldorf.sciebo.de/s/P5l3VKXyKEV7QSb/download)

[1] Reimers, N., & Gurevych, I. (2019, November). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 3982-3992).

Code to create pre-training data and run pre-training from: https://github.com/google-research/bert

Used settings for data creation:

Setting|Value
---|---
vocab_file|uncased_L-8_H-512_A-8/vocab.txt
do_lower_case|True
max_seq_length|128
max_predictions_per_seq|5
masked_lm_prob|0.15
randon_seed|12345
dupe_factor|10

Used settings for pre-training:

Setting|Value
---|---
BERT model|BERT medium uncased, BERT base uncased
learning rate|1e-4, 2e-5
max_seq_length|128
max_predictions_per_seq|5
train_batch_size|32
eval_batch_size|8
