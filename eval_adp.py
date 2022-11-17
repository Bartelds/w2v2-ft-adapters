import jiwer
import torch
import os
import soundfile as sf
import omegaconf as oc
import pandas as pd
import numpy as np

from datasets import Dataset
from tqdm import tqdm

from helpers import (
    utils,
    w2v2_apt
)

config = oc.OmegaConf.from_cli()

assert '--config' in config.keys(), """\n
    Please supply a base config file, e.g. 'python train.py --config=CONFIG_FILE.yml'.
    You can then over-ride config parameters, e.g. 'python train.py --config=CONFIG_FILE.yml trainargs.learning_rate=1e-5'
"""

config, wandb_run = utils.make_config(config)

w2v2_config = {
    "bottleneck_adapters_kwargs" : {
        "use_bottleneck_adapter": True,
        "bottleneck_adapter_dim" : 256,
        "bottleneck_adapter_act" : "gelu",
        "unfreeze_layernorm" : True,
        "unfreeze_encoder": False
    },
    "cnn_adapters_kwargs": {
        "use_cnn_adapter": False,
        "cnn_adapter_do_norm": True,
        "cnn_adapter_kernel": 1,
        "cnn_adapter_stride": 1
    }
}

utils.announce("Configuring model")

model, processor = w2v2_apt.configure_hf_w2v2_model(config, w2v2_config)
model = model.eval().cuda()

devset_path = os.path.join(config['data'].base_path, config['data'].train_tsv)
testset_path = os.path.join(config['data'].base_path, config['data'].eval_tsv)

print(f"Development set: {devset_path}")
print(f"Test set: {testset_path}")

dev_ds = pd.read_csv(devset_path, sep = '\t')
test_ds = pd.read_csv(testset_path, sep = '\t')

def _read_audio(path):
    full_path = os.path.join(config['data'].base_path, path)

    data, sr = sf.read(full_path)

    assert sr == 16_000

    return data

dev_ds['audio'] = [ _read_audio(path) for path in tqdm(dev_ds['path'].to_list(), desc='Reading audio data') ]
test_ds['audio'] = [ _read_audio(path) for path in tqdm(test_ds['path'].to_list(), desc='Reading audio data') ]

dev_ds = Dataset.from_pandas(dev_ds[['audio', 'text']])
test_ds = Dataset.from_pandas(test_ds[['audio', 'text']])

utils.announce('Evaluating model')

def evaluate(batch):
    inputs = processor(batch['audio'], sampling_rate=16_000, return_tensors='pt', padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits

    pred_ids = np.argmax(logits.cpu(), axis=-1)
    batch['transcription'] = processor.batch_decode(pred_ids)
    
    return batch

dev_ds = dev_ds.map(evaluate, batched=True, batch_size=1)
test_ds = test_ds.map(evaluate, batched=True, batch_size=1)

wer_dev = round(jiwer.wer(dev_ds['text'], dev_ds['transcription']), 4)
cer_dev = round(jiwer.cer(dev_ds['text'], dev_ds['transcription']), 4)

wer_test = round(jiwer.wer(test_ds['text'], test_ds['transcription']), 4)
cer_test = round(jiwer.cer(test_ds['text'], test_ds['transcription']), 4)

utils.announce('Results on development data')

print(f"WER: {wer_dev} CER: {cer_dev}")

utils.announce('Results on test data')

print(f"WER: {wer_test} CER: {cer_test}")
