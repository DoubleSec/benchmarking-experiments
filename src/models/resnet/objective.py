import torch
from morphers import Integerizer, Normalizer
from lightning import pytorch as pl

from .net import ResNet
from .data import ResNetDataset


# Until it get fixed in morphers
class Normalizer(Normalizer):

    def fill_missing(self, x):
        return self.backend.fill_missing(x, self.mean)


def objective(trial, data, cfg):

    # Set up model
    features = [(feat, "numeric", {}) for feat in cfg.features.numeric] + [
        (feat, "categorical", {}) for feat in cfg.features.categorical
    ]

    # TKTK Configurable morphers
    morpher_map = {
        "numeric": Normalizer,
        "categorical": Integerizer,
    }

    # Yuck
    features = [(feature, (morpher_map[t], d)) for (feature, t, d) in features]
    n_features = len(features)

    train_ds = ResNetDataset(data["train_data"], features)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
    )
    validation_ds = ResNetDataset(data["train_data"], features, train_ds.morphers)
    validation_dl = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
    )

    n_categories = data["train_data"]["target"].max()
    loss_func = (
        torch.nn.BCEWithLogitsLoss if n_categories == 1 else torch.nn.CrossEntropyLoss
    )

    output_dim = n_categories + 1

    resnet_params = {
        # Fixed
        "loss_func": loss_func,
        "output_dim": output_dim,
        # From optuna
        "embedding_dim": trial.suggest_int("embedding_dim", 1, 32),
        "n_layers": trial.suggest_int("n_layers", 1, 8),
        "dropout": trial.suggest_float("dropout", 0, 0.5),
        "optimizer_args": {
            "lr": trial.suggest_float("learning_rate", 0.00001, 0.001, log=True),
        },
        "morpher_dict": train_ds.morphers,
    }
    resnet_params["hidden_dim"] = resnet_params["embedding_dim"] * n_features

    torch.manual_seed(cfg.model.seed)

    net = ResNet(**resnet_params)

    trainer = pl.Trainer(
        accelerator=cfg.model.device,
        precision=cfg.model.precision,
        max_epochs=cfg.model.n_epochs,
        log_every_n_steps=cfg.model.log_every_n_steps,
    )

    trainer.fit(net, train_dl, validation_dl)
    return trainer.callback_metrics["validation_AUROC"]
