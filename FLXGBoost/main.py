import warnings
from pathlib import Path
import pickle
import pandas
import hydra
import xgboost as xgb
import flwr as fl
from flwr.server.strategy import FedXgbBagging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from dataSet import prepare_dataset
from utils import (
    sim_args_parser,
    NUM_LOCAL_ROUND,
    BST_PARAMS,
)
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    aggregated_fit
)
from client_utils import XgbClient

warnings.filterwarnings("ignore", category=UserWarning)


def get_client_fn(
        trainloaders, valloaders, train_method, params, num_local_round
):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        # Zugriff auf die Daten im trainloader für den Client mit der ID cid
        train_iterator = iter(trainloaders[int(cid)])
        x_train, y_train = next(train_iterator)

        # Zugriff auf die Daten im valloader für den Client mit der ID cid
        val_iterator = iter(valloaders[int(cid)])
        x_valid, y_valid = next(val_iterator)

        # Reformat data to DMatrix
        train_dmatrix = xgb.DMatrix(x_train.numpy(), label=y_train.numpy())
        valid_dmatrix = xgb.DMatrix(x_valid.numpy(), label=y_valid.numpy())

        # Fetch the number of examples
        num_train = len(trainloaders[int(cid)].dataset)
        num_val = len(valloaders[int(cid)].dataset)

        # Create and return client
        return XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
        )

    return client_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Parse arguments for experimental settings
    args = sim_args_parser()
    save_path = HydraConfig.get().runtime.output_dir

    # Load and partition your dataset using the prepare_dataset function
    trainloaders, valloaders, testloader = prepare_dataset()

    # Define strategy
    strategy = FedXgbBagging(
        evaluate_function=(get_evaluate_fn(testloader)
                           ),
        fraction_fit=(float(args.num_clients_per_round) / args.pool_size),
        min_fit_clients=args.num_clients_per_round,
        min_available_clients=args.pool_size,
        min_evaluate_clients=(
            args.num_evaluate_clients if not args.centralised_eval else 0
        ),
        fraction_evaluate=1.0 if not args.centralised_eval else 0.0,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not args.centralised_eval else None
        ),
        fit_metrics_aggregation_fn= aggregated_fit
    )

    # Resources to be assigned to each virtual client
    # In this example we use CPU by default
    client_resources = {
        "num_cpus": args.num_cpus_per_client,
        "num_gpus": 0.0,
    }

    # Hyper-parameters for xgboost training
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS

    # Setup learning rate
    if args.train_method == "bagging" and args.scaled_lr:
        new_lr = params["eta"] / args.pool_size
        params.update({"eta": new_lr})

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(
            trainloaders,
            valloaders,
            args.train_method,
            params,
            num_local_round,
        ),
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_manager=None,
    )

    # 6. Save your results
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


def showResults():
    result = pandas.read_pickle("../outputs/2024-05-14/11-58-32/results.pkl")
    print(result)


if __name__ == "__main__":
    main()
    #showResults()
