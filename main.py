
from train import train
from predict import predict
from conllu_to_jsonl import read_josn_srl
import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv
    else:
        None
    

    read_josn_srl("data/original data/en_ewt-up-train.conllu")
    read_josn_srl("data/original data/en_ewt-up-test.conllu")
    train_file = "data/preprocessed data/train.jsonl"
    dev_file = "data/preprocessed data/test.jsonl"
    train(train_file, dev_file)
    predict(dev_file)

        
        
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    