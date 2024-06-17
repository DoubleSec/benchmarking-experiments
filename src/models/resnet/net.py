import torch
from torch import nn
from lightning import pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC


class ResNetBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float):

        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):

        return x + self.layers(x)


class ResNet(pl.LightningModule):
    # TKTK configurable norms/activations

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        morpher_dict: dict,
        loss_func: type,
        optimizer_args: dict,
    ):
        super().__init__()

        assert hidden_dim % embedding_dim == 0

        self.save_hyperparameters()

        # Paranoia about feature order
        self.feature_order = list(morpher_dict.keys())

        self.morpher_dict = morpher_dict

        self.embedding_layers = nn.ModuleDict(
            {
                feature: morpher.make_embedding(embedding_dim)
                for feature, morpher in self.morpher_dict.items()
            }
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

        # Set up loss and optimizer
        self.loss = loss_func(reduction="none")
        self.optim_args = optimizer_args

        # Logging
        # TKTK support multi-label, because this doesn't.
        task = "binary" if isinstance(self.loss, nn.BCEWithLogitsLoss) else "multiclass"
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task=task, average="weighted", num_classes=output_dim
                ),
                "AUROC": AUROC(task=task, average="weighted", num_classes=output_dim),
            }
        )
        # Make a metric collection for train and validation
        self.train_metrics = metrics.clone(prefix="train_")
        self.validation_metrics = metrics.clone(prefix="validation_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optim_args)
        return optimizer

    def forward(self, x: dict):

        x = [self.embedding_layers[feat](x[feat]) for feat in self.feature_order]

        x = torch.cat(x, dim=-1)
        x = self.input_norm(x)
        x = self.blocks(x)
        x = self.prediction_head(x)

        return x

    def step(self, x):
        preds = self(x)
        loss = self.loss(preds, x["target"]).mean()
        return loss, preds

    def training_step(self, x):
        loss, preds = self.step(x)
        self.log_dict(self.train_metrics(preds, x["target"]), prog_bar=True)
        return loss

    def validation_step(self, x):
        loss, preds = self.step(x)
        self.log_dict(self.validation_metrics(preds, x["target"]), prog_bar=True)
        return self.step(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # TKTK small norm init for bias, too
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
