import os
import jiwer
import jsonlines
from uie_utils import get_fpr
import argparse

def get_wer_from_text(file_path):
    predictions=[]
    references=[]
    with open(file_path,"r",encoding="UTF-8") as f:
        line=f.readline()
        while line:
            prediction,reference=line[:-1].split("<->")
            predictions.append(prediction)
            references.append(reference)
            line=f.readline()

    w=jiwer.wer(references,predictions)*100
    return w

def get_fpr_from_text(file_path):
    predictions=[]
    references=[]
    with open(file_path, 'r') as f:
        for data in jsonlines.Reader(f):
            if data["pred_entities"]:
                for pred in data["pred_entities"]:
                    predictions.append(pred)
            if data["true_entities"]:
                for true in data["true_entities"]:
                    references.append(true)
    f,p,r=get_fpr(predictions,references)
    return f,p,r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True, help='path to inference result')
    args = parser.parse_args()

    output_path=args.output_path
    dataset_name=output_path.split("/")[-1].split("|")[1]
    if dataset_name=="AISHELL-NER":
        cer=get_wer_from_text(os.path.join(output_path,"transcript.txt"))
        f1,precision,recall=get_fpr_from_text(os.path.join(output_path,"structure.jsonl"))
        print(f"AISHELL-NER cer:{cer}, f1:{f1}, precision:{precision}, recall:{recall}")
    elif dataset_name=="SLURP":
        wer=get_wer_from_text(os.path.join(output_path,"transcript.txt"))
        print(f"SLURP wer:{wer}\nPlease use SLURP's official inference code to get NER and IC result.")
    else:
        raise Exception(f"unknown dataset:{args.dataset_name}")
