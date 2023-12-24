# Generative Emotion Cause Triplet Extraction in Conversations with Commonsense Knowledge

Here are codes for our paper: 

*Fanfan Wang, Jianfei Yu and Rui Xia. [Generative Emotion Cause Triplet Extraction in Conversations with Commonsense Knowledge](https://aclanthology.org/2023.findings-emnlp.260). Findings of the Association for Computational Linguistics: EMNLP 2023. 2023: 3952-3963.*

## Environment

- Python 3.8.15
- Cuda 11.3 and Nvidia RTX-3090 GPU
- Run `pip install -r requirements.txt` to install the required packages.

## Usage

Training:

```
python train.py  --save_model 1  --save_path ./output/SHARK/
```

Testing:

```
python test.py  --seed 2023  --save_path ./output/SHARK/  --model_file ./output/SHARK/save_models/2023/xxxxxx
```

## Acknowledgements

- Some codes are based on [BARTABSA](https://github.com/yhcc/BARTABSA). Thanks a lot!

