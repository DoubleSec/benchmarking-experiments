import polars as pl
import torch
import yaml


def load_morphers(
    state_path: str,
    cols: dict,
    type_map: dict,
):
    with open(state_path, "r") as f:
        morpher_states = yaml.load(f, Loader=yaml.CLoader)

    morphers = {
        col: type_map[ctype].from_state_dict(morpher_states[col])
        for col, ctype in cols.items()
    }
    return morphers


class ResNetDataset(torch.utils.data.Dataset):

    def __init__(self, df: pl.DataFrame, cols: list, morphers: dict = None):
        super().__init__()

        # Use existing morpher states
        if morphers is None:
            morphers = {
                feature: morpher_class.from_data(df[feature], **kwargs)
                for feature, (morpher_class, kwargs) in cols
            }
        else:
            morphers = morphers

        self.data = df.select(
            *[
                morpher(morpher.fill_missing(pl.col(feature))).alias(feature)
                for feature, morpher in morphers.items()
            ],
            "target"
        )
        self.morphers = morphers

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):

        row = self.data.row(idx, named=True)
        return {
            k: torch.tensor(row[k], dtype=morpher.required_dtype)
            for k, morpher in self.morphers.items()
        } | {"target": torch.tensor(row["target"])}
