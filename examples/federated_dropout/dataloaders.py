from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from flwr_datasets import FederatedDataset


class CustomDataset(Dataset):
    """
    Convert huggingface dataset to pytorch dataset
    """
    def __init__(self, dataset, name, split):
        self.dataset = dataset
        self.name = name
        self.split = split
        self.transform = self.get_transforms(name, split)

    @staticmethod
    def get_transforms(name, split):
        transform_dict = {
            "cifar10": {
                "train": transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
                "test": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            },
            "mnist": {
                "train": transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                "test": transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1325,), (0.3105,))
                ])
            }
        }

        if name not in transform_dict:
            raise ValueError(f"Unsupported dataset name. Supported names are {tuple(transform_dict.keys())}.")
        if split not in transform_dict[name]:
            raise ValueError(f"Unsupported split mode. Supported modes are {tuple(transform_dict[name].keys())}.")

        return transform_dict[name][split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_field_dict = {"cifar10": 'img', "mnist": 'image'}
        data_field = data_field_dict[self.name]

        item = self.dataset[idx]
        img = self.transform(item[data_field])
        label = item['label']
        return img, label


def load_partition_data(dataset_name, partitioner, cid, batch_size=32):
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

    train_set = fds.load_partition(partition_id=cid-1, split="train")  # datasets.arrow_dataset.Dataset
    train_partition = CustomDataset(train_set, name=dataset_name, split="train")
    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)

    test_set = fds.load_split(split="test")  # datasets.arrow_dataset.Dataset
    test_partition = CustomDataset(test_set, name=dataset_name, split="test")
    test_loader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)

    num_examples = {"train_set": len(train_set), "test_set": len(test_set)}
    return train_loader, test_loader, num_examples
