import pandas as pd
import json

def read_conll_file(file_path):
    '''read original conllu file and return all sentences'''
    sentences = []
    with open(file_path, encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if not line:
                sentences.append(current_sentence)
                current_sentence = []
            elif not line.startswith('#'):
                fields = line.split('\t')
                current_sentence.append(fields)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def repeat_multi_predicate_sentences(sentences):
    '''split each sentence into propositions based on predicates'''
    final_sentences = []
    for sentence in sentences:

        # Check if sentence has at least 11 fields
        if len(sentence[0]) < 11:
            continue
        # Get the values of all predicate columns in the sentence
        predicate_values = list([fields[10] for fields in sentence if len(fields) >= 11])
        predicate_values = [item for item in predicate_values if item != '_']
        # If there is only one predicate value or no predicates, don't repeat the sentence
        if len(predicate_values) <= 1:
            final_sentences.append(sentence)
        elif len(predicate_values) > 1:
            # Repeat sentence for each predicate value
            for i, pred in enumerate(predicate_values):
                b = i + 1
                # Convert sentence to DataFrame
                df = pd.DataFrame(sentence)
                # Create a new DataFrame with only the first 11 columns of the original DataFrame
                df_2 = df.iloc[:, :11].copy()
                new_col = df.iloc[:, (10 + b)]
                df_2[11] = new_col
                new_sentence = df_2.values.tolist()
                final_sentences.append(new_sentence)
                
    return final_sentences

def read_josn_srl(inputfile):
    '''write all preprocessed sentences to corresponding jsonl files and save in the specified path'''
    sentences = read_conll_file(inputfile)
    final_sentences = repeat_multi_predicate_sentences(sentences)
    if inputfile == 'data/original data/en_ewt-up-train.conllu':
        outfile = 'data/preprocessed data/train.jsonl'
    if inputfile == 'data/original data/en_ewt-up-test.conllu':
        outfile = 'data/preprocessed data/test.jsonl'
    with open(outfile, 'w') as outputfile:
        for i in final_sentences:
            tokens = []
            bio = []
            pred_sense = []
            # loop over sentences and add tokens to a list inside of a dict per sentence
            for j in i:
                sentence_dict = {}
                tokens.append(j[1])
                # if element at index 11 is a dash, append an O, otherwise append the element with B- in front of it.
                if len(j) != 12:
                    continue
                if j[11] is None or j[11] == "_":
                    bio.append("O")
                else:
                    bio.append("B-"+j[11])
                if len(j) >= 12 and j[11] == "V":
                    pred_sense.append(int(j[0])-1)
                    pred_sense.append(j[10])
                    pred_sense.append(j[11])
                    pred_sense.append(j[4])
            # add everything to the dict per sentence
            sentence_dict['seq_words'] = tokens
            sentence_dict['BIO'] = bio
            sentence_dict['pred_sense'] = pred_sense
            # write the sentence dictionary to the output file as a JSON object
            outputfile.write(json.dumps(sentence_dict) + '\n')
            
read_josn_srl('data/original data/en_ewt-up-train.conllu')
read_josn_srl('data/original data/en_ewt-up-test.conllu')