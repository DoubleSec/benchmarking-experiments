from typing import Callable, Tuple
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_bs(
    test_data: pl.DataFrame, model: Callable[[pl.DataFrame], np.ndarray]
) -> Tuple[float]:
    """Evaluate a model for the balls and strikes task.

    - test_data: a polars dataframe containing input data and targets. This will be created in advance.
    - model: a callable, which takes a polars dataframe and returns an numpy ndarray of p(strike), NOT logits.

    Returns: a float indicating the area under the ROC curve for this model on the test data.
    """

    targets = test_data["target"].to_numpy()
    inputs = test_data.drop("target")
    preds = model(inputs)

    return roc_auc_score(targets, preds), average_precision_score(targets, preds)


def evaluate_pt(
    test_data: pl.DataFrame, model: Callable[[pl.DataFrame], np.ndarray]
) -> float:
    """Evaluate a model for the pitch type.

    - test_data: a polars dataframe containing input data and targets. This will be created in advance.
    - model: a callable, which takes a polars dataframe and returns an numpy ndarray of p(strike), NOT logits.

    Note the model's output should be [len(test_data), n_classes] in shape. By default n_classes is 8.

    Returns: a float indicating the area under the ROC curve for this model on the test data.
    """

    targets = test_data["target"].to_numpy()
    inputs = test_data.drop("target")
    preds = model(inputs)

    # bruh no
    return roc_auc_score(
        targets, preds, average="weighted", multi_class="ovr"
    ), average_precision_score(targets, preds, average="weighted")


if __name__ == "__main__":

    # Obviously this all belongs in a test.

    from scipy.special import softmax

    def guessing_bs_model(test_data):

        rng = np.random.default_rng()
        return rng.random([test_data.shape[0]])

    def guessing_pt_model(test_data):

        rng = np.random.default_rng()
        logits = rng.random([test_data.shape[0], 8])
        return softmax(logits, axis=1)

    # Test balls and strikes eval
    test_data = pl.read_parquet("./data/bs_test.parquet")
    score = evaluate_bs(test_data, guessing_bs_model)
    print(score)

    # Test pitch_type eval
    test_data = pl.read_parquet("./data/pt_test.parquet")
    score = evaluate_pt(test_data, guessing_pt_model)
    print(score)
