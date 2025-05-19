import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dakshina_data import DakshinaDataModule
from model import Seq2SeqModel


def train(
    embed_dim,
    hidden_dim,
    n_layers,
    cell_type,
    dropout,
    learning_rate,
    data_dir,
    lang_code,
    max_len,
    num_workers,
    epochs,
    use_attention,
    batch_size=32,
    wandb_project=None,
    run_name=None,
    test=True,
):
    pl.seed_everything(42)
    torch.cuda.empty_cache()

    # Datamodule
    datamodule = DakshinaDataModule(
        data_dir=data_dir,
        lang_code=lang_code,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
    )
    datamodule.setup()

    # Trainer
    wandb_logger = WandbLogger(log_model="all", project=wandb_project, name=run_name)
    wandb_logger.experiment.name = run_name + "-" + wandb_logger.experiment.name

    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", mode="max", min_delta=0.01, patience=3
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=20,
        val_check_interval=200,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Model
    model = Seq2SeqModel(
        source_vocab_size=datamodule.source_tokenizer.vocab_size,
        target_vocab_size=datamodule.target_tokenizer.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        cell_type=cell_type,
        dropout=dropout,
        learning_rate=learning_rate,
        use_attention=use_attention,
    )

    trainer.fit(model, datamodule=datamodule)
    if test:
        trainer.test(model, datamodule=datamodule)
