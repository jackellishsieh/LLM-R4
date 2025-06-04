"""
This file contains curriculum dataset definitions
"""
import json
import random
from datasets import Dataset
from abc import ABC, abstractmethod

class CurriculumDataset(ABC, Dataset):
    """
    Base class for selection-based curriculum datasets.
    This class extends the `datasets.Dataset` class and provides a mechanism for updating the selection of examples.
    The step function is to be implemented to change the active indices of the dataset.
    """

    def __init__(self, dataset_path: str, dataset_size: int = 1024):
        # Load entire JSON once (e.g. a list of dicts)
        with open(dataset_path, "r") as file:
            initial_data = json.load(file)

        # Process the list of dictionaries as appropriate
        self._all_data = self.process_json(initial_data)
        self.active_indices = None
        self.dataset_size = dataset_size
        return

    def __len__(self):
        assert self.active_indices is not None, "active_indices must be set through step() before accessing items."
        return len(self.active_indices)

    def __getitem__(self, idx: int):
        assert self.active_indices is not None, "active_indices must be set through step() before accessing items."

        # Map “idx in [0... len(active_indices))” to the real example
        real_idx = self.active_indices[idx]
        example_dict = self._all_data[real_idx]
        return example_dict

    @abstractmethod
    def process_json(self, initial_data: list[dict]) -> list[dict]:
        """
        Process the JSON file and return a list of dictionaries.
        This method should be implemented by subclasses to define how the JSON data is processed.
        """
        raise NotImplementedError("Subclasses must implement the process_json method.")

    @abstractmethod
    def step(self):
        """
        Update the active_indices for the dataset, to determine which examples to use in the next epoch.
        """
        raise NotImplementedError("Subclasses must implement the step method to update active_indices.")
    

class VanillaDataset(CurriculumDataset):
    def __init__(self, dataset_path: str, dataset_size: int = 1024):
        """
        Initializes the VanillaDataset with a JSON file path and an optional dataset size limit at any given epoch.
        :param json_path: Path to the JSON file containing the source dataset.
        :param dataset_size: Optional limit on the number of examples to include in the dataset at any given epoch.
        """
        super().__init__(dataset_path, dataset_size)

        # Initialize remaining indices in random order
        self.remaining_indices = random.shuffle(list(range(len(self._all_data))))

        return

    def process_json(self, initial_data: list[dict]) -> list[dict]:
        # Does no extra processing, just returns the data as is
        return initial_data

    def step(self):
        assert len(self.remaining_indices) < self.dataset_size, "Not enough remaining indices to sample a batch."

        # Activate the next set of indices for the dataset, remove these from remaining
        self.active_indices = self.remaining_indices[:self.dataset_size]
        self.remaining_indices = self.remaining_indices[self.dataset_size:]
        return