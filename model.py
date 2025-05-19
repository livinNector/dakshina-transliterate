from typing import Literal
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, MetricCollection


class BaseModule(pl.LightningModule):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size

        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
                "f1_score": F1Score(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
            },
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
                "f1_score": F1Score(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
            },
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
                "f1_score": F1Score(
                    "multiclass", num_classes=self.vocab_size, average="macro"
                ),
            },
            prefix="test_",
        )

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, logger=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        # query: (B, 1, H)
        # keys: (B, T, H)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (B, T, 1)
        scores = scores.squeeze(2).unsqueeze(1)  # (B, 1, T)

        weights = F.softmax(scores, dim=-1)  # attention weights (B, 1, T)
        context = torch.bmm(weights, keys)  # weighted sum (B, 1, H)

        return context, weights


class Seq2SeqModel(BaseModule):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        embed_dim=128,
        hidden_dim=256,
        n_layers=1,
        cell_type: Literal["RNN", "LSTM", "GRU"] = "LSTM",
        use_attention: bool = False,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(vocab_size=target_vocab_size)
        self.save_hyperparameters()

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cell_type = cell_type.upper()
        self.dropout = dropout
        self.use_attention = use_attention

        self.encoder_embedding = nn.Embedding(source_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embed_dim)

        rnn_cls = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }[self.cell_type]

        self.encoder = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.use_attention = use_attention

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim)
            # Decoder input size now is embed_dim + hidden_dim (due to concat with context)
            self.decoder = rnn_cls(
                input_size=embed_dim + hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.decoder = rnn_cls(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
            )
        self.output_fc = nn.Linear(hidden_dim, target_vocab_size)

        self.learning_rate = learning_rate

    def forward(self, source, target):
        batch_size, target_len = target.size()

        embedded_source = self.encoder_embedding(source)  # (B, S, E)
        enc_outputs, hidden = self.encoder(embedded_source)  # (B, S, H)

        if self.use_attention:
            dec_hidden = hidden
            embedded_target = self.decoder_embedding(target)  # (B, T, E)

            outputs = []
            for t in range(target_len):
                dec_input = embedded_target[:, t : t + 1, :]  # (B, 1, E)

                if self.cell_type == "LSTM":
                    query = dec_hidden[0][-1:].permute(1, 0, 2)  # (B, 1, H)
                else:
                    query = dec_hidden[-1:].permute(1, 0, 2)  # (B, 1, H)

                context, attn_weights = self.attention(query, enc_outputs)  # (B, 1, H)

                rnn_input = torch.cat([dec_input, context], dim=2)  # (B, 1, E + H)

                dec_output, dec_hidden = self.decoder(rnn_input, dec_hidden)

                logits = self.output_fc(dec_output)  # (B, 1, V)
                outputs.append(logits)

            outputs = torch.cat(outputs, dim=1)  # (B, T, V)
            return outputs

        else:
            # Original forward (no attention)
            embedded_target = self.decoder_embedding(target)  # (B, T, E)
            dec_outputs, _ = self.decoder(embedded_target, hidden)
            logits = self.output_fc(dec_outputs)  # (B, T, V)
            return logits

    def training_step(self, batch, batch_idx):
        source, target = batch
        logits = self(source, target[:, :-1])  # Predict next token
        target_shifted = target[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, self.target_vocab_size),
            target_shifted.reshape(-1),
            ignore_index=0,
        )
        self.log("train_loss", loss)

        preds = torch.argmax(logits, dim=-1)  # (B, T)
        mask = target_shifted != 0  # Ignore padding
        self.train_metrics.update(preds[mask], target_shifted[mask])

        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch
        logits = self(source, target[:, :-1])
        target_shifted = target[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, self.target_vocab_size),
            target_shifted.reshape(-1),
            ignore_index=0,
        )
        self.log("val_loss", loss)

        preds = torch.argmax(logits, dim=-1)
        mask = target_shifted != 0
        self.val_metrics.update(preds[mask], target_shifted[mask])

    def test_step(self, batch, batch_idx):
        source, target = batch
        logits = self(source, target[:, :-1])
        target_shifted = target[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, self.target_vocab_size),
            target_shifted.reshape(-1),
            ignore_index=0,
        )
        self.log("test_loss", loss)

        preds = torch.argmax(logits, dim=-1)
        mask = target_shifted != 0
        self.test_metrics.update(preds[mask], target_shifted[mask])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
