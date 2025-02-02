import os
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import random
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
from torch.utils.data import dataset, Sampler
import sys
import itertools

# Hyperparameters
NJ = 16
TARGET_SAMPLE_RATE = 16000


class ASRdataset(Dataset):
    def __init__(self, data, stoi, sp):
        self.data = data
        self.stoi = stoi
        self.sp = sp
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.slice(idx, 1)
        row_data = {col:row.column(col).to_pylist()[0] for col in row.schema.names}
        audio, sr = torchaudio.load(row_data["path"])

        if sr != TARGET_SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            audio = resampler(audio)
            # print(audio.shape)

        if audio.shape[0]>1:
            c = random.randint(0, audio.shape[0]-1)
            audio = audio[c].unsqueeze(0)
            
        tokens = [self.stoi.get(t, self.stoi["<unk>"]) for t in self.sp.EncodeAsPieces(row_data["text"]) + ["<eos>"]]
        return {"id":row_data["id_"] ,"speech":audio, "length":audio.shape[-1], "tokens":torch.LongTensor(tokens)}
    


def prepare_text_vocab(train_set, dump_dir):
    if not os.path.exists(os.path.join(dump_dir, "spm.model")) or not os.path.exists(os.path.join(dump_dir, "spm.vocab")):
        os.makedirs(dump_dir, exist_ok=True)
        dump_text_path = os.path.join(dump_dir, "dump_text.txt")
        
        all_text = train_set["text"].to_pylist()
        with open(dump_text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        
        spm.SentencePieceTrainer.train(
            input=dump_text_path,
            model_prefix=os.path.join(dump_dir, "spm"), 
            model_type="bpe",
            vocab_size=300,
            character_coverage=1.0,
            user_defined_symbols=["<sos>", "<eos>"]
        )
    
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(dump_dir, "spm.model"))
    
    with open(os.path.join(dump_dir, "spm.vocab"), "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()
    
    stoi = {line.split()[0]: i+1 for i, line in enumerate(vocab)}
    itos = {v: k for k, v in stoi.items()}
    return sp, stoi, itos


def prepare_datasets(data_path, train_set_name, valid_set_name, expdir):
    
    # TODO don't delete old experiments
    # expdir = Path(expdir)
    # if expdir.exists():
        
    #     print(f"experiment dir already exists : {expdir}. Renaming it to {}")
    #     shutil.rmtree(expdir)
    os.makedirs(expdir, exist_ok=True)
    
    data_path = Path(data_path)
    
    train_set = pq.read_table((data_path / train_set_name).with_suffix(".parquet"))
    valid_set = pq.read_table((data_path / valid_set_name).with_suffix(".parquet"))
    
    for name in ["id_", "duration", "path", "text"]:
        assert name in train_set.schema.names
    
    dump_dir = os.path.join(expdir, "dump")
    sp, stoi, itos = prepare_text_vocab(train_set, dump_dir)
    
    train_set = ASRdataset(train_set, stoi, sp)
    valid_set = ASRdataset(valid_set, stoi, sp)

    return train_set, valid_set, stoi, itos, sp


class SortedSampler(Sampler[list[int]]):
    def __init__(self, data_source, max_frames, batch_size, seed, stft_center, win_length, hop_length, rank, world_size):
        self.data_source = data_source
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        
        def get_frame_length(dur):
            ilen = dur * TARGET_SAMPLE_RATE
            if stft_center:
                olen = 1 + ilen // hop_length
            else:
                olen = 1 + (ilen - win_length) // hop_length
            return olen
        
        print(f"Setting up Sorted batch sampler with max frame length : {max_frames} and max batch size : {batch_size}")
        indices = {k:int(get_frame_length(dur)) for k, dur in enumerate(self.data_source.data.column("duration").to_pylist())}
        indices = dict(sorted(indices.items(), key=lambda item: item[1], reverse=True))
        batches = []
        batch = []
        batch_length = 0
        # print(indices)
        for i, len_ in indices.items():
            if batch_length + len_ <= max_frames and len(batch)<batch_size:
                batch.append(i)
                batch_length += len_
            else:
                batches.append(batch)
                batch = [i]
                batch_length = len_
        

        random.shuffle(batches)
        batches_per_gpu = len(batches) // world_size
        self.batches = batches[rank*batches_per_gpu : (rank+1)*batches_per_gpu]
        
    
    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

class UnsortedSampler(Sampler[list[int]]):
    def __init__(self, data_source, max_frames, batch_size, seed, stft_center, win_length, hop_length, rank, world_size):
        self.data_source = data_source
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        print(f"Setting up Unsorted batch sampler with fixed batch size: {batch_size}")

        total_samples = len(self.data_source.data.column("duration"))
        indices = list(range(total_samples))
        random.shuffle(indices)

        batches = [indices[i : i + batch_size]for i in range(0, len(indices), batch_size)]
        batches_per_gpu = len(batches) // world_size
        self.batches = batches[rank * batches_per_gpu : (rank + 1) * batches_per_gpu]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    

def collate_fn(batch):
    ids_ = [b["id"] for b in batch]
    speech = [b["speech"].squeeze(0) for b in batch]
    tokens = [b["tokens"] for b in batch]
    lengths = [b["length"] for b in batch]
    
    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True, padding_value=0.0)
    text = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    
    return {
        "ids_": ids_,
        "speech": speech,
        "tokens": text,
        "lengths": lengths
    }
    

