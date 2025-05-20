from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch


class CharTokenizer:
    def __init__(self, special_tokens=["<pad>", "<sos>", "<eos>"]):
        self.special_tokens = special_tokens
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)

        all_tokens = self.special_tokens + sorted(list(chars))
        self.char2idx = {char: idx for idx, char in enumerate(all_tokens)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, text, add_sos=False, add_eos=False):
        tokens = []
        if add_sos:
            tokens.append(self.char2idx[self.sos_token])
        tokens += [
            self.char2idx.get(char, self.char2idx[self.pad_token]) for char in text
        ]
        if add_eos:
            tokens.append(self.char2idx[self.eos_token])
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        chars = []
        for idx in token_ids:
            char = self.idx2char.get(idx, "")
            if skip_special_tokens and char in self.special_tokens:
                if char == self.eos_token:
                    break
                continue
            chars.append(char)
        return "".join(chars)

    def pad_sequence(self, sequence, max_len):
        return sequence[:max_len] + [self.char2idx[self.pad_token]] * max(
            0, max_len - len(sequence)
        )

    def batch_encode(self, texts, add_sos=False, add_eos=False, max_len=None):
        encoded = [self.encode(text, add_sos, add_eos) for text in texts]
        if not max_len:
            max_len = max(len(seq) for seq in encoded)
        encoded = [self.pad_sequence(seq, max_len) for seq in encoded]
        return encoded

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]


class DakshinaDataset(Dataset):
    def __init__(self, tsv_path, to_en=False):
        self.data = []

        with tsv_path.open(encoding="utf-8") as f:
            self.data = [
                parts[:2] if to_en else parts[:2][::-1] + [int(parts[2])]
                for line in f
                if len(parts := line.strip().split("\t")) >= 3
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data[idx]
        else:
            return self.data[idx]


class DakshinaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        lang_code,
        to_en=False,
        batch_size=32,
        max_len=128,
        num_workers=2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.lang_code = lang_code
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.to_en = to_en

    def setup(self, stage=None):
        lex_path = self.data_dir / self.lang_code / "lexicons"
        self.train_ds = DakshinaDataset(
            lex_path / f"{self.lang_code}.translit.sampled.train.tsv", to_en=self.to_en
        )
        self.val_ds = DakshinaDataset(
            lex_path / f"{self.lang_code}.translit.sampled.dev.tsv", to_en=self.to_en
        )
        self.test_ds = DakshinaDataset(
            lex_path / f"{self.lang_code}.translit.sampled.test.tsv", to_en=self.to_en
        )
        source_texts, target_texts = (
            [item[0] for item in self.train_ds],
            [item[1] for item in self.train_ds],
        )
        self.source_tokenizer = CharTokenizer()
        self.target_tokenizer = CharTokenizer()
        self.source_tokenizer.build_vocab(source_texts)
        self.target_tokenizer.build_vocab(target_texts)

    def collate_fn(self, batch):
        source_batch = [item[0] for item in batch]
        target_batch = [item[1] for item in batch]

        source_encoded = torch.tensor(
            self.source_tokenizer.batch_encode(
                source_batch, add_eos=True, add_sos=True, max_len=self.max_len
            )
        )

        target_encoded = torch.tensor(
            self.target_tokenizer.batch_encode(
                target_batch, add_eos=True, add_sos=True, max_len=self.max_len
            )
        )

        return source_encoded, target_encoded

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
