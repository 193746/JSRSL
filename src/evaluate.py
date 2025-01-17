import os
import datetime
# from modelscope.metainfo import Trainers
# from modelscope.trainers import build_trainer
# from modelscope.msdatasets.audio.asr_dataset import ASRDataset
from asr_trainer import ASRTrainer
from asr_dataset import ASRDataset


def modelscope_finetune(params):
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    asr_trainer=ASRTrainer(model=params.model,
                           data_dir=ds_dict,
                           dataset_type=params.dataset_type,
                           batch_bins=params.batch_bins,
                           max_epoch=params.max_epoch,
                           lr=params.lr,
                           dataset_name=params.dataset_name)
    asr_trainer.evaluate()

if __name__ == '__main__':

    from funasr.utils.modelscope_param import modelscope_args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='path to model')
    parser.add_argument('--dataset_name', type=str, choices=["AISHELL-NER", "SLURP"], default="SLURP", help='dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='path to training data')
    parser.add_argument('--dataset_type', type=str, default="small")
    parser.add_argument('--batch_bins', type=int, default=4000, help='fbank frames')
    args = parser.parse_args()

    params = modelscope_args(model=args.model_name)
    params.dataset_name = args.dataset_name
    params.data_path = args.data_path  
    params.dataset_type = args.dataset_type
    params.batch_bins = args.batch_bins
    modelscope_finetune(params)