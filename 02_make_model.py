from functools import partial

from src.models import xgboost, resnet
import optuna
import hydra
import polars as pl

CONFIGURED_MODELS = {
    "xgboost": xgboost.objective,
    "resnet": resnet.objective,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    # Get data and subset to correct features
    training_data = pl.read_parquet(f"{cfg.destination}/{cfg.task}_train.parquet")
    training_data = training_data.select(
        *[cfg.features.numeric + cfg.features.categorical + ["target"]]
    )

    # Shuffle training data and split into test and validation
    train_frac = 0.8
    max_train_idx = int(train_frac * len(training_data))
    training_data = training_data.sample(fraction=1, seed=cfg.seed, shuffle=True)

    train_data = training_data[:max_train_idx]
    validation_data = training_data[max_train_idx:]

    # TKTK logging
    print(f"Training: {train_data.height} observations")
    print(f"Validation: {validation_data.height} observations")

    data_dict = {
        "train_data": train_data,
        "validation_data": validation_data,
    }

    objective = CONFIGURED_MODELS[cfg.model.model_name]
    obj = partial(objective, data=data_dict, cfg=cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=cfg.n_trials)

    print(study.best_params)


if __name__ == "__main__":
    main()
