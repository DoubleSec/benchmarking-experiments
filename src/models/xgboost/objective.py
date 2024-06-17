from xgboost import XGBClassifier


def objective(trial, data, cfg):

    # Set up model
    xgb_params = {
        # Fixed
        "early_stopping_rounds": 50,
        "enable_categorical": True,
        "eval_metric": "auc",
        "multi_strategy": "one_output_per_tree",
        # From config
        "device": cfg.model.device,
        "sampling_method": cfg.model.sampling_method,
        "random_state": cfg.model.seed,
        # From optuna
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 2, 30),
    }

    # Set up data
    pd_data = {
        "x_train": data["train_data"].to_pandas().drop("target", axis=1),
        "y_train": data["train_data"]["target"].to_numpy(),
        "x_validation": data["validation_data"].to_pandas().drop("target", axis=1),
        "y_validation": data["validation_data"]["target"].to_numpy(),
    }

    model = XGBClassifier(**xgb_params)

    eval_set = [
        (pd_data["x_train"], pd_data["y_train"]),
        (pd_data["x_validation"], pd_data["y_validation"]),
    ]

    model.fit(pd_data["x_train"], pd_data["y_train"], eval_set=eval_set, verbose=30)

    return model.best_score
