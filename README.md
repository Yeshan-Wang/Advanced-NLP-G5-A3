# Advanced-NLP-G5-A3
The code was carried out by Payam Fakhraie and Yeshan Wang during the course ‘Advanced NLP' taught by Luis Morgado da Costa and Jose Angel Daza at VU Amsterdam.

## REQUIREMENTS
- torch==1.7.1 
- keras==2.6.0
- tensorflow==2.6.0
- transformers==4.9.1
- datasets==1.11.0
- tabulate==0.8.9
- seqeval==1.2.2

## conllu_to_jsonl.py
The script splits each sentence into propositions based on predicates from the original data (so each instance has a single labeled argument structure) and save as corresponding json files in the data directory:
- data/preprocessed data/train.jsonl
- data/preprocessed data/test.jsonl

## train.py
The script which is used to fine-tune the BERT model on the training set.

## predict.py
The script loads an already fine-tuned model produced by train.py and makes predictions on the test set.

## bert_utils.py


## main.py
The main() function in main.py is the entry point for executing the program. When the main() function is called, it performs the following steps:
- Calls the read_josn_srl() function twice to read the data from two files, "en_ewt-up-train.conllu" and "en_ewt-up-test.conllu".
- Specifies the paths of the preprocessed data files, "train.jsonl" and "test.jsonl", and calls the train() function to train the model using the training data and validate it using the test data.
- Calls the predict() function to predict the labels of the test data.
