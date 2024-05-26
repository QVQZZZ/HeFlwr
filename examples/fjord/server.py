import flwr as fl

from strategy import Fjord


# To track the acc during training (using federation evaluation).
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5),
                       strategy=Fjord(evaluate_metrics_aggregation_fn=weighted_average))
