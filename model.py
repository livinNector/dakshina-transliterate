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
        beam_width: int | None = None,
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
        self.beam_width = beam_width

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

    def greedy_predict(self, source, sos_idx=1, max_len=128):
        """
        Greedy decoding for inference.
        If self.use_attention is True, performs attention-based decoding.
        Otherwise, uses standard decoder without attention.

        Returns:
            predictions: (B, max_len) tensor of predicted token indices
        """
        batch_size = source.size(0)
        device = source.device

        embedded_source = self.encoder_embedding(source)
        enc_outputs, hidden = self.encoder(embedded_source)

        if self.cell_type == "LSTM":
            dec_hidden = (hidden[0].clone(), hidden[1].clone())
        else:
            dec_hidden = hidden.clone()

        input_token = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=device
        )
        embedded_input = self.decoder_embedding(input_token)

        predictions = []

        for i in range(max_len):
            if self.use_attention:
                if self.cell_type == "LSTM":
                    query = dec_hidden[0][-1:].permute(1, 0, 2)  # (B, 1, H)
                else:
                    query = dec_hidden[-1:].permute(1, 0, 2)  # (B, 1, H)

                context, _ = self.attention(query, enc_outputs)  # (B, 1, H)
                rnn_input = torch.cat([embedded_input, context], dim=2)  # (B, 1, E+H)
            else:
                rnn_input = embedded_input  # (B, 1, E)

            dec_output, dec_hidden = self.decoder(rnn_input, dec_hidden)
            logits = self.output_fc(dec_output)  # (B, 1, V)
            next_token = logits.argmax(-1)  # (B, 1)

            predictions.append(next_token)

            embedded_input = self.decoder_embedding(next_token)

        predictions = torch.cat(predictions, dim=1)  # (B, max_len)
        return predictions

    def beam_search_predict(
        self, source, sos_idx=1, eos_idx=2, beam_width=None, max_len=128
    ):
        """
        Batched beam search decoding.

        Args:
            source: (B, S) input tensor
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_width: number of beams per input
            max_len: max decoding length

        Returns:
            List[List[int]]: Decoded token indices for each input
        """
        batch_size = source.size(0)
        device = source.device
        beam_width = beam_width or self.beam_width or 3

        embedded_source = self.encoder_embedding(source)  # (B, S, E)
        enc_outputs, hidden = self.encoder(embedded_source)

        if self.cell_type == "LSTM":
            h, c = hidden
        else:
            h = hidden

        results = []

        for i in range(batch_size):
            enc_out_i = enc_outputs[i : i + 1]  # (1, S, H)
            if self.cell_type == "LSTM":
                h_i = h[:, i : i + 1, :].clone()  # (num_layers, 1, H)
                c_i = c[:, i : i + 1, :].clone()
                hidden_i = (h_i, c_i)
            else:
                hidden_i = h[:, i : i + 1, :].clone()

            beams = [
                {
                    "tokens": [sos_idx],
                    "log_prob": 0.0,
                    "hidden": hidden_i,
                    "embedded_input": self.decoder_embedding(
                        torch.tensor([[sos_idx]], device=device)
                    ),
                }
            ]
            completed = []

            for _ in range(max_len):
                new_beams = []

                for beam in beams:
                    last_token = beam["tokens"][-1]
                    if last_token == eos_idx:
                        completed.append(beam)
                        continue

                    embedded_input = beam["embedded_input"]
                    hidden = beam["hidden"]

                    if self.use_attention:
                        if self.cell_type == "LSTM":
                            query = hidden[0][-1:].permute(1, 0, 2)
                        else:
                            query = hidden[-1:].permute(1, 0, 2)

                        context, _ = self.attention(query, enc_out_i)
                        rnn_input = torch.cat([embedded_input, context], dim=2)
                    else:
                        rnn_input = embedded_input

                    dec_output, new_hidden = self.decoder(rnn_input, hidden)
                    logits = self.output_fc(dec_output)  # (1, 1, V)
                    log_probs = F.log_softmax(logits.squeeze(1), dim=-1)  # (1, V)

                    topk_log_probs, topk_indices = torch.topk(
                        log_probs, beam_width, dim=-1
                    )

                    for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                        idx = idx.item()
                        log_prob = log_prob.item()

                        new_tokens = beam["tokens"] + [idx]
                        new_log_prob = beam["log_prob"] + log_prob
                        new_embedded = self.decoder_embedding(
                            torch.tensor([[idx]], device=device)
                        )
                        new_beams.append(
                            {
                                "tokens": new_tokens,
                                "log_prob": new_log_prob,
                                "hidden": (
                                    (new_hidden[0].clone(), new_hidden[1].clone())
                                    if self.cell_type == "LSTM"
                                    else new_hidden.clone()
                                ),
                                "embedded_input": new_embedded,
                            }
                        )

                beams = sorted(new_beams, key=lambda x: x["log_prob"], reverse=True)[
                    :beam_width
                ]

                if len(completed) >= beam_width:
                    break

            if not completed:
                completed = beams

            best_beam = max(completed, key=lambda x: x["log_prob"])
            results.append(best_beam["tokens"])

        return results

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

    def predict_step(self, batch, batch_idx):
        source, *_ = batch
        preds = self.beam_search_predict(source)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
