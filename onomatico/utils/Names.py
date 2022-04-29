import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, build_vocab_from_iterator

from pathlib import Path
from typing import Iterator, Optional, Tuple
from numpy import ndarray
import pandas as pd
from pandas import DataFrame


class NamesDataset(Dataset):
    unk_token = "?"
    start_token = "<"
    stop_token = ">"
    padding_token = "!"
    names: ndarray
    padded_sequence_length: int
    """Load names dataset and provide them as bached tensor of specific size"""

    def __init__(
        self, csv_file: Path, name_column: str = "name", vocab: Optional[vocab] = None
    ):
        """Load the names in given csv file

        Args:
            csv_file (Path): _description_
            name_column (str, optional): _description_. Defaults to "name".
        """
        self.csv_file = csv_file
        data = pd.read_csv(csv_file)
        self.names = data[name_column].values
        self.max_sequence_length = data[name_column].str.len().max()
        self.padded_sequence_length = self.max_sequence_length + 2
        self.__create_tokens(vocab)

    def __create_tokens(self, vocab: Optional[vocab] = None):
        """Build a vocab based on the characters in `self.names`
        Build a list of equally sized tensors encoding the names with special
        start, stop and padding token.
        """
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = build_vocab_from_iterator(
                self.names,
                specials=[
                    NamesDataset.unk_token,
                    NamesDataset.start_token,
                    NamesDataset.stop_token,
                    NamesDataset.padding_token,
                ],
            )
            self.vocab.set_default_index(self.vocab[NamesDataset.unk_token])
        # self.names_tks = [torch.tensor(self.vocab(chr), dtype=torch.long) for chr in name for name in self.names]
        self.names_tks = []
        for name in self.names:
            # Each name is translated into a sequence of the same length containing a start, stop and padding tokens if needed
            # Example: Bob Miller -> <Bob Miller>!!!
            n = list(
                f"{NamesDataset.start_token}{name}{NamesDataset.stop_token}".ljust(
                    self.padded_sequence_length, NamesDataset.padding_token
                )
            )
            self.names_tks.append(torch.tensor(self.vocab(n), dtype=torch.long))

    def __len__(self) -> int:
        return len(self.names_tks)

    def __getitem__(self, idx: int) -> torch.tensor:
        return self.names_tks[idx]


class Names(Iterator):
    names_dataset: NamesDataset
    names_dl: DataLoader
    names_iter: Iterator
    device: torch.device
    batch_size: int

    def __init__(self, csv_file: Path, batch_size: int, device: torch.device, **param):
        """Build a torch DataLoader reading from given csv_file

        Args:
            csv_file (Path): Path to csv file containing names
            batch_size (int): Batch size
            device (torch.device): torch device
        """
        self.names_dataset = NamesDataset(csv_file, **param)
        self.names_dl = DataLoader(
            self.names_dataset, batch_size=batch_size, shuffle=True
        )
        self.names_iter = iter(self.names_dl)
        self.batch_size = batch_size

    def get_padded_sequence_length(self) -> int:
        return self.names_dataset.padded_sequence_length

    def get_batch(self) -> Tuple[torch.tensor, torch.tensor]:
        """Returns a mini-batch of names, with the target is shifted by one position.

        Returns:
            Tuple[torch.tensor, torch.tensor]: Mini-batch of (training, target)
        """
        # Get a mini-batch of encoded name token sequences
        batch = next(self.names_iter)

        # The target is shifted by one element
        # data: "<Bob MILLER>!!",
        # target: "Bob MILLER>!!!"
        data = batch[:, :-1]
        target = batch[:, 1:]
        return data, target

    def __len__(self) -> int:
        return len(self.names_dl)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Tuple[torch.tensor, torch.tensor]:
        """Wrap the iterator generated from the DataLoader.
        Make sure to get refresh the iterator once it is emptied.
        So for a single epoch we reach the stop token, but can read
        from the iteration again in the next epoch.

        Raises:
            e: StopIteration

        Returns:
            Tuple[torch.tensor, torch.tensor]: Training and Label data 
        """
        try:
            return self.get_batch()
        except StopIteration as e:
            self.names_iter = iter(self.names_dl)
            raise e
