from typing import Tuple, List, Union, Optional
from typing_extensions import Self
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Layer_Range


class SSEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, 
                 norm_type: float = 2.0, 
                 scale_grad_by_freq: bool = False, 
                 sparse: bool = False,
                 embedding_dim_ranges: Layer_Range = ('0', '1')) -> None:

        # if features_ranges belong to Interval, then convert into Intervals.
        if isinstance(embedding_dim_ranges[0], str):
            embedding_dim_ranges = [embedding_dim_ranges]

        # Convert string interval into fraction interval.
        embedding_dim_ranges = [tuple(map(self.parse_fraction_strings, range_str)) for range_str in embedding_dim_ranges]

        # pruned embedding_dim
        pruned_embedding_dim = sum(int(embedding_dim * (end - start)) for start, end in embedding_dim_ranges)

        super(SSEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=pruned_embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

        self.embedding_dim_ranges = embedding_dim_ranges
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input, self.weight, self.padding_idx, 
            self.max_norm, self.norm_type, 
            self.scale_grad_by_freq, self.sparse
        )

    def extra_repr(self) -> str:
        base_str = super().extra_repr()
        return f'{base_str}, embedding_dim_ranges={self.embedding_dim_ranges}'

    def reset_parameters_from_father_layer(self, father_layer: nn.Embedding) -> None:
        """
        Resets the parameters of the current layer based on the parameters of a parent (father) layer.

        This method is used to propagate parameters from a parent layer to the current layer, effectively
        initializing or updating the current layer's parameters.

        The method identifies matching blocks of parameters between the layers based on their relative
        positions, and uses the `get_subset_parameters` and `set_subset_parameters` methods to transfer
        the parameters.

        :param father_layer: The parent layer of type nn.Embedding
        from which the parameters are to be propagated.

        :return: None. The method updates the parameters of the current layer in place.
        """
        father_dim_indices_start = [int(range_[0] * father_layer.embedding_dim) for range_ in self.embedding_dim_ranges]
        father_dim_indices_end = [int(range_[1] * father_layer.embedding_dim) for range_ in self.embedding_dim_ranges]
        father_dim_indices = list(zip(father_dim_indices_start, father_dim_indices_end))

        child_dim_indices = self.convert_indices(father_dim_indices)

        with torch.no_grad():
            for father_idx, child_idx in zip(father_dim_indices, child_dim_indices):
                weight = self.get_subset_parameters(father_layer, father_idx)
                self.set_subset_parameters(weight, child_idx)

    @staticmethod
    def get_subset_parameters(father_layer: nn.Embedding, dim_index: Tuple[int, int]) -> torch.Tensor:
        return father_layer.weight[:, dim_index[0]: dim_index[1]].clone()

    def set_subset_parameters(self, weight: torch.Tensor, dim_index: Tuple[int, int]) -> None:
        self.weight[:, dim_index[0]: dim_index[1]] = weight

    @staticmethod
    def convert_indices(indices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        ret_indices = []
        offset = 0
        for start, end in indices:
            ret_start = offset
            ret_end = offset + (end - start)
            ret_indices.append((ret_start, ret_end))
            offset = ret_end
        return ret_indices

    @staticmethod
    def parse_fraction_strings(fraction_str: str) -> Fraction:
        if fraction_str == '0':
            return Fraction(0, 1)
        if fraction_str == '1':
            return Fraction(1, 1)
        numerator, denominator = map(int, fraction_str.split('/'))
        return Fraction(numerator, denominator)
