from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from flwr_datasets import FederatedDataset


class CustomDataset(Dataset):
    """
    Convert huggingface dataset to pytorch dataset
    """
    def __init__(self, dataset, name, split):
        self.dataset = dataset
        if name == "cifar10":
            if split == "train":
                trans = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif split == "test":
                trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                raise ValueError("Unsupported split mode. Supported modes are 'train' and 'test'.")
        elif name == "mnist":
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            raise ValueError("Unsupported dataset name. Supported names are 'mnist' and 'cifar10'.")
        self.transform = trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.transform(item['img'])
        label = item['label']
        return img, label


def load_partition_data(dataset_name, partitioner, cid, batch_size=32):
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

    train_set = fds.load_partition(partition_id=cid, split="train")  # datasets.arrow_dataset.Dataset
    train_partition = CustomDataset(train_set, name=dataset_name, split="train")
    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)

    test_set = fds.load_split(split="test")  # datasets.arrow_dataset.Dataset
    test_partition = CustomDataset(test_set, name=dataset_name, split="test")
    test_loader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)

    num_examples = {"train_set": len(train_set), "test_set": len(test_set)}
    return train_loader, test_loader, num_examples
