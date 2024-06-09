from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def load_data(path="../data", name="cifar10", batch_size=32):
    if name == "cifar10":
        """Load CIFAR-10 (training and test_monitors.py set)."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = CIFAR10(root=path, train=True, download=True, transform=transform_train)
        test_set = CIFAR10(root=path, train=False, download=True, transform=transform_test)
    elif name == "mnist":
        """Load MNIST (training and test_monitors.py set)."""
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset name. Supported names are 'mnist' and 'cifar10'.")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    num_examples = {"train_set": len(train_set), "test_set": len(test_set)}
    return train_loader, test_loader, num_examples
