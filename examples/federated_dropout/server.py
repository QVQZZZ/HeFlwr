import argparse
import flwr as fl

from lenet import LeNet
from resnet import ResNet18
from strategy import FederatedDropout


# To track the acc during training (using federation evaluation).
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heflwr baseline server.")
    parser.add_argument('--dataset', type=str, help='Dataset name.')
    parser.add_argument('--num_rounds', type=int, help='Total rounds of fl.')
    args = parser.parse_args()
    dataset = args.dataset
    num_rounds = args.num_rounds

    if dataset == "cifar10":
        network = ResNet18
    elif dataset == "mnist":
        network = LeNet

    history = fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds),
                                     strategy=FederatedDropout(evaluate_metrics_aggregation_fn=weighted_average, network=network))
    print(history)