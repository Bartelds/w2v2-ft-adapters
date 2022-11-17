import omegaconf as oc
import transformers as hft

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

print(config)

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

# Number of trainable parameters
print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')
print(f'Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print(f'Trainable adapter parameters: {sum(p.numel() for n,p in model.named_parameters() if "bottleneck_adapter" in n)}')
print(f'Trainable CNN adapter parameters: {sum(p.numel() for n, p in model.named_parameters() if "cnn_adapter" in n)}')

datasets = utils.load_datasets(config['data'], processor)

# Set verbosity to info, otherwise trainer progress bar isn't shown
hft.logging.set_verbosity_info()

trainer = w2v2_apt.ReplicationTrainer(
    model=model,
    data_collator=w2v2_apt.DataCollatorCTCWithPadding(processor=processor, padding=True),
    args=hft.TrainingArguments(**config['trainargs']),
    compute_metrics=w2v2_apt.MetricsComputer(config, processor),
    train_dataset=datasets['train'],
    eval_dataset=datasets['eval'],
    tokenizer=processor.feature_extractor
)

utils.announce("Beginning training")

trainer.train()
