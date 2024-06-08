from typing import Dict, List, Optional, Tuple
from logging import INFO
import xgboost as xgb
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from sklearn.metrics import accuracy_score, precision_score, f1_score
from FLXGBoost.utils import BST_PARAMS


def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])

    # Initialisierung der aggregierten Metriken
    auc_aggregated = 0.0
    accuracy_aggregated = 0.0
    precision_aggregated = 0.0
    f1_aggregated = 0.0

    # Berechnung der gewichteten Durchschnittswerte
    for num, metrics in eval_metrics:
        auc_aggregated += metrics["AUC"] * num
        accuracy_aggregated += metrics["Accuracy"] * num
        precision_aggregated += metrics["Precision"] * num
        f1_aggregated += metrics["F1 Score"] * num


    # Berechnung der Durchschnittswerte
    auc_aggregated /= total_num
    accuracy_aggregated /= total_num
    precision_aggregated /= total_num
    f1_aggregated /= total_num

    # Zusammenfassung der aggregierten Metriken
    metrics_aggregated = {
        "AUC": auc_aggregated,
        "Accuracy": accuracy_aggregated,
        "Precision": precision_aggregated,
        "F1 Score": f1_aggregated,
    }

    return metrics_aggregated


def get_evaluate_fn(testloader):
    """Return a function for centralised evaluation."""
    X_test, y_test = next(iter(testloader[0]))
    test_dmatrix = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())

    def evaluate_fn(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=BST_PARAMS)
            for para in parameters.tensors:
                para_b = bytearray(para)
            bst.load_model(para_b)

            eval_results = bst.eval_set(
                evals=[(test_dmatrix, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            preds = bst.predict(test_dmatrix)
            labels = test_dmatrix.get_label()
            preds_binary = [1 if pred > 0.5 else 0 for pred in preds]

            accuracy = accuracy_score(labels, preds_binary)
            precision = precision_score(labels, preds_binary)
            f1 = f1_score(labels, preds_binary)

            log(INFO,
                f"AUC = {auc}, Accuracy = {accuracy}, Precision = {precision}, F1 Score = {f1} at round {server_round}")

            return 0, {
                "AUC": auc,
                "Accuracy": accuracy,
                "Precision": precision,
                "F1 Score": f1
            }
    return evaluate_fn


def aggregated_fit(
        rnd: int, results: List[Tuple[float, Parameters]]
) -> Tuple[int, Optional[Parameters]]:
    global GLOBAL_MODEL

    # Weighted average of model updates
    total_examples = sum(num_examples for num_examples, _ in results)
    new_weights = []
    for num_examples, parameters in results:
        weight = num_examples / total_examples
        new_weights.append((weight, parameters))

    # Initialize aggregated model
    aggregated_model = None

    # Aggregate models using weighted average
    for weight, parameters in new_weights:
        model = xgb.Booster(params=BST_PARAMS)
        for para in parameters.tensors:
            para_b = bytearray(para)
        model.load_model(para_b)

        # If this is the first model, initialize the aggregated model
        if aggregated_model is None:
            aggregated_model = model
        else:
            # Update the aggregated model by combining the current model with the previous aggregated model
            aggregated_model = aggregated_model.update(
                model,
                aggregation="mean"
            )

    # Save the aggregated model
    aggregated_model_bytes = aggregated_model.save_raw("json")

    # Update the global model
    GLOBAL_MODEL = Parameters(
        tensor_type="", tensors=[bytes(aggregated_model_bytes)]
    )

    return 0, GLOBAL_MODEL