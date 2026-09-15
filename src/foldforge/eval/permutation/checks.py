# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Checker:
    """Represent checker."""

    @staticmethod
    def is_permutation(x: torch.Tensor) -> None:
        """Check if the input tensor `x` is a permutation of integers from 0 to N-1.

        Args:
            x (torch.Tensor): A 1D tensor of size [N].
        """
        if x.dim() != 1:
            message = "Invalid state: x.dim() == 1"
            raise ValueError(message)
        N = x.size(0)
        if not (torch.equal(torch.sort(x)[0], torch.arange(N, device=x.device))):
            message = (
                "Invalid state: torch.equal(torch.sort(x)[0], torch.arange(N, "
                "device=x.device))"
            )
            raise ValueError(message)

    @staticmethod
    def are_permutations(x: torch.Tensor, dim: int = -1) -> None:
        """Compute are permutations.

        Check if slices along the specified dimension in `x` are permutations of
        integers from 0 to N-1.

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices
                of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations.
                Defaults to -1.
        """
        if not (x.dim() > 0):
            message = "Invalid state: x.dim() > 0"
            raise ValueError(message)

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        torch.arange(N, device=x.device)
        for i in range(x.size(0)):
            Checker.is_permutation(x[i])

    @staticmethod
    def contains_identity(x: torch.Tensor, dim: int = -1) -> None:
        """Check if x contains the identity permutation.

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices
                of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations.
                Defaults to -1.
        """
        if not (x.dim() > 0):
            message = "Invalid state: x.dim() > 0"
            raise ValueError(message)

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        if not ((x == expected).all(dim=-1).any()):
            message = "Invalid state: (x == expected).all(dim=-1).any()"
            raise ValueError(message)

    @staticmethod
    def not_contain_identity(x: torch.Tensor, dim: int = -1) -> None:
        """Check if x does not contain the identity permutation.

        Args:
            x (torch.Tensor): A tensor with any number of dimensions, containing slices
                of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations.
                Defaults to -1.
        """
        if not (x.dim() > 0):
            message = "Invalid state: x.dim() > 0"
            raise ValueError(message)

        N = x.size(dim)
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, N)
        expected = torch.arange(N, device=x.device).unsqueeze(dim=0)
        if (x == expected).all(dim=-1).any():
            message = "Invalid state: not (x == expected).all(dim=-1).any()"
            raise ValueError(message)

    @staticmethod
    def batch_permute(
        perm: torch.Tensor, x: torch.Tensor, x_permuted: torch.Tensor
    ) -> None:
        """Args:

        perm (torch.Tensor):
            [..., N]
        x (torch.Tensor):
            [N, batch_dims_x]
        x_permuted (torch.Tensor):
            [..., N, batch_dims_x]
        """
        batch_shape = perm.shape[:-1]
        N = perm.size(-1)
        if x.size(0) != N:
            message = "Invalid state: x.size(0) == N"
            raise ValueError(message)
        perm = perm.view(-1, N)
        permuted_x = [x[perm[i]] for i in range(len(perm))]
        permuted_x = torch.stack(permuted_x, dim=0)  # [-1, N, batch_dims_x]
        target_shape = (*batch_shape, N, *x.shape[1:])
        if not (torch.allclose(permuted_x.reshape(target_shape), x_permuted)):
            message = (
                "Invalid state: torch.allclose(permuted_x.reshape(target_shape), "
                "x_permuted)"
            )
            raise ValueError(message)


def save_permutation_error(
    data, error_dir: str | None = None, max_cases: int = 50
) -> None:
    """Save the permutation error data to a specified directory.

    Args:
        data: The data to be saved.
        error_dir (str): The directory where the error data should be saved.
        max_cases (int): The maximum number of error cases to save.

    Raises:
        Exception: If an error occurs while saving the data, the exception is caught
            and printed.
    """
    if error_dir is None:
        return

    # error_dir = os.path.join(self.error_dir, dir_name)
    Path(error_dir).mkdir(parents=True, exist_ok=True)

    if sum(1 for _ in Path(error_dir).iterdir()) >= max_cases:
        # Only record the first {max_cases} error cases for debug
        return

    filename = "T_" + time.strftime("%Y%m%d_%H%M%S") + ".pt"
    fpath = str(Path(error_dir) / (filename))
    if not Path(fpath).exists():
        try:
            torch.save(data, fpath)
        except Exception:
            logger.exception("Failed to save permutation diagnostic")
