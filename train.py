import os
from data_utils import prepare_datasets
from E2asr_model import ASRconfig, E2ASR
from trainer import Trainer, set_seed
import wandb
import torch


# HYPER PARAMETERS
data_path = "/speech/shoutrik/torch_exp/E2asr/data/LibriTTS"
expdir = "/speech/shoutrik/torch_exp/E2asr/exp/LibriTTS_trial05"
train_set_name = "train"
valid_set_name = "dev_clean"
max_frames = 64000
batch_size = 128
max_epoch = 200
grad_norm_threshold = 1.0
save_last_step_freq = 2000
save_global_step_freq = 40000
logging_freq = 100
seed=42
accum_grad=3
learning_rate = 2e-4
warmup_steps = 40000
weight_decay=0.1

config = ASRconfig(
    sample_rate= 16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=80,
    center=True,
    preemphasis=True,
    normalize_energy=False,
    time_mask_param=30,
    freq_mask_param=15,
    norm_mean=True,
    norm_var=True,
    model_dim=768,
    feedforward_dim=3072,
    dropout=0.1,
    num_heads=12,
    num_layers=24,
    max_len=4000,
    stochastic_depth_p=0.1,
    unskipped_layers=[0,1,2,3,4,5,21,22,23],
)


# logging related
wandb_project="E2asr"
wandb_run_name=os.path.basename(expdir)
run_id=wandb.util.generate_id()
resume="allow"


set_seed(42)

train_dataset, valid_dataset, stoi, itos, sp = prepare_datasets(data_path, train_set_name, valid_set_name, expdir)
vocab_size = len(stoi) + 1
model = E2ASR(config, vocab_size, training=True)


trainer = Trainer(model=model,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  max_frames=max_frames,
                  batch_size=batch_size,
                  config=config,
                  expdir=expdir,
                  accum_grad=accum_grad,
                  max_epoch=max_epoch,
                  idx_to_char_map=itos,
                  save_last_step_freq=save_last_step_freq,
                  save_global_step_freq=save_global_step_freq,
                  resume_from_checkpoint=False,
                  logging_freq=logging_freq,
                  grad_norm_threshold=grad_norm_threshold,
                  seed=seed,
                  learning_rate=learning_rate,
                  warmup_steps=warmup_steps,
                  weight_decay=weight_decay,
                  step_to_start_layer_drop=20000,
                  logger="wandb",
                  wandb_project=wandb_project,
                  wandb_run_name=wandb_run_name,
                  run_id=run_id,
                  resume=resume,
                  dataloader_num_workers=4,
                  ddp=False,
                  )

trainer.train()
wandb.finish()
