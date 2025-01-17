## [Joint Automatic Speech Recognition And Structure Learning For Better Speech Understanding](https://arxiv.org/abs/2410.00822)
This repository is the official implementation of JSRSL. The paper has been accepted by ICASSP 2025.
### Prepare dataset
Download the audio data from AISHELL-NER or SLURP and place them in JSRSL/dataset/{dataset}/audio
```sh
JSRSL/
│
└── dataset/
    ├── AISHELL-NER/
    │   ├── audio/
    |   |   ├── dev/
    |   |   ├── test/
    |   |   └── train/
    │   └── train_data/   
    └── SLURP/
        ├── audio/
        |   ├── real/
        |   ├── synth/
        └── train_data/  

```
JSRSL/dataset/{dataset}/train_data/{split}/wav.scp record the required audio path.

The audios of AISHELL-NER are available at https://www.openslr.org/33/.

The audios of SLURP are available at https://github.com/pswietojanski/slurp.

### Install packages
```sh
pip install -r requirements.txt
```

### Train
Download the base model. Please download the Paraformer's Chinese version: "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" if you train on the AISHELL-NER and download the Paraformer's English version: "speech_paraformer_asr-en-16k-vocab4199-pytorch" when training on the SLURP. Taking training on the SLURP as an example.

```sh
from modelscope import snapshot_download
snapshot_download('damo/speech_paraformer_asr-en-16k-vocab4199-pytorch',local_dir='{path_to_save_model}')
```

Copy the model file and othe config files to JSRSL/pretrained_model/JSRSL_SLURP
```sh
cp -rn {path_to_save_model}/speech_paraformer_asr-en-16k-vocab4199-pytorch/* JSRSL/pretrained_model/JSRSL_SLURP
```

Download the pretrained refiner and put it in JSRSL/pretrained_model/JSRSL_SLURP/bert. Our refiner in English and Chinese versions are available at [Hugging Face](https://huggingface.co/Rinawell/JSRSL).

```sh
JSRSL/
│
└── pretrained_model/
    ├── JSRSL_SLURP/
    |   ├── bert
    |   |   ├── config.json
    |   |   ├── pytorch_model.bin
    |   |   └── tokens.txt
    |   ├── am.mvn
    |   ├── config.yaml
    |   ├── configuration.json
    |   ├── decoding.yaml
    |   ├── finetune.yaml
    |   ├── model.pb
    |   ├── seg_dict
    |   └── tokens.txt
    └─── JSRSL_AISHELL-NER/
```

Start training.
```sh
cd JSRSL
CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
--model_name "pretrained_model/JSRSL_{dataset}" \
--output_dir "ft_JSRSL_{dataset}_checkpoint" \
--dataset_name "{dataset}" \
--data_path "dataset/{dataset}/train_data" \
--batch_bins 4000 \
--epoch 80 \
--lr 5e-5
```

### Test
After training, place the trained model file and other configuration files in the same folder for subsequent testing.

```sh
cd JSRSL
CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
--model_name "pretrained_model/JSRSL_{dataset}" \
--dataset_name "{dataset}" \
--data_path "dataset/{dataset}/train_data" \
--batch_bins 4000
```

Our trained JSRSL is available at [Hugging Face](https://huggingface.co/Rinawell/JSRSL).

The inference result will be save in the "output/{date-time}|{dataset}|{model_name}".
```sh
JSRSL/
│
└── output/
    └── 01-17-21-19-56|SLURP|JSRSL_SLURP/
        ├── structure.jsonl
        └── transcript.txt
```

For AISHELL-NER, you can use the following code to obtain the result.

```sh
cd JSRSL
python src/get_result.py \
--output_path "output/{output_directory}"
```

For SLURP, the above code only gets WER. You should use SLURP's official inference code to get SLURP-F1 and other metrics. Please download the [SLURP](https://github.com/pswietojanski/slurp) and install it's requirements, then evaluate as follows.
```sh
cd slurp/scripts/evaluation
python evaluate.py \
-g "slurp/dataset/slurp/test.jsonl" \
-p "JSRSL/output/{output_directory}/structure.jsonl"
```

### Statement
Most of the code in this repository is modified from https://github.com/modelscope/FunASR/tree/v0.8.8 

### Citation
```sh
@misc{hu2025jointautomaticspeechrecognition,
      title={Joint Automatic Speech Recognition And Structure Learning For Better Speech Understanding}, 
      author={Jiliang Hu and Zuchao Li and Mengjia Shen and Haojun Ai and Sheng Li and Jun Zhang},
      year={2025},
      eprint={2501.07329},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.07329}, 
}
```

### License: cc-by-nc-4.0