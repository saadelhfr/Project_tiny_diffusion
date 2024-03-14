from typing import List, Tuple
from src.data.CombinedData import CombinedDataset
from src.utils.datasets import get_dataset
from torch.utils.data import Dataset, ConcatDataset


class DataSetTinyDiffusion(Dataset):
    def __init__(
        self,
        number_data_points,
        device,
        joined,
        custom_dataset=None,
        name_dataset=None,
    ):
        super(DataSetTinyDiffusion, self).__init__()
        self.name_dataset = name_dataset
        self.number_data_points = number_data_points
        self.device = device
        self.custom_dataset = custom_dataset
        self.joined = joined
        if self.joined and custom_dataset is not None:
            assert isinstance(
                custom_dataset[0], CombinedDataset
            ), "for adding a custom dataset make sure it is of the class CombinedData. The definition of the class is in src/data/CombinedData.py"

        self.initialise_data()  # adds self.data_points to the class

    def __len__(self):
        return len(self.finale_dataset)

    def __getitem__(self, index):

        return self.finale_dataset[index]

    def initialise_data(self):
        self.data_points1 = []
        self.data_points2 = []
        if self.name_dataset is not None:
            if isinstance(self.name_dataset, List):
                for name in self.name_dataset:
                    if self.joined:
                        self.data_points1.append(
                            CombinedDataset(
                                self.get_data_from_name(name),
                                self.get_data_from_name(name),
                            )
                        )
                    else:
                        self.data_points1.append(self.get_data_from_name(name))

            else:
                if self.joined:
                    self.data_points1.append(
                        CombinedDataset(
                            self.get_data_from_name(self.name_dataset),
                            self.get_data_from_name(self.name_dataset),
                        )
                    )
                else:
                    self.data_points1.append(self.get_data_from_name(self.name_dataset))
        if self.custom_dataset is not None:
            self.data_points2 = self.custom_dataset
        self.data_points1.extend(self.data_points2)
        print(self.data_points1)
        self.finale_dataset = ConcatDataset(self.data_points1)

    def get_data_from_name(self, name):
        assert isinstance(
            name, Tuple
        ), "Expected 'name' to be a tuple of (name, path), got something else. The names should be tuples of a name and a path , if the dataset is one of 'moons' , 'dino' , 'line' , 'circle' provide a None object for the path"
        name_str = name[0]
        path = name[1]
        if name_str == "mnist":
            return get_dataset(name=name_str, n=self.number_data_points, path=path)
        elif name_str == "moons":
            return get_dataset(name=name_str, n=self.number_data_points)
        elif name_str == "dino":
            return get_dataset(name=name_str, n=self.number_data_points)
        elif name_str == "line":
            return get_dataset(name=name_str, n=self.number_data_points)
        elif name_str == "circle":
            return get_dataset(name=name_str, n=self.number_data_points)
        else:
            raise ValueError(
                f"Unknown dataset : {name_str}, If you wanna use a custom dataset provide it with the argument custom_dataset which should be a list you can also provide additiona dataset names to concatenate all the data"
            )
