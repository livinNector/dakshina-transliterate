# dakshina-transliterate
Implementation of recurrent and attention based networks on dakshina dataset.

# Code Organization

- **Data Module** - The `dakshina_data.py` contains the datamodule used which data loaders for different splits of the data.
- **Model** - The `model.py` contains the seq2seq model implementation with different recurrent units and optional attention.
- **Trainer** - The `trainer.py` contains the `train` function which configures all the necessary things for a training run.
- **CLI** - The `train.py` provides a command line interface to the `train` function. This is meant to start a run from cli all parameters can be found using `python train.py --help`
- **Sweep Configs** - The `sweep.yaml` and `sweep_attention.yaml` were used to configure wandb sweeps using the wandb cli.

Hyperparameter tuning is done by calling the wandb agent from cli.

# Prediction

The predictions from best models done in the `predict.ipynb` the test dataset is used for this and the resulting predictions are saved in `predictions_vannila.csv` and `predictions_attention.csv` files. The Sample prediction table is plotted using plotly and saved to wandb.