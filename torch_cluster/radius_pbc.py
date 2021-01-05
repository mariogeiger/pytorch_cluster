from typing import Optional

import ase.neighborlist
import torch


def radius_graph_pbc(x: torch.Tensor, r: float, cell: torch.Tensor,
                     batch: Optional[torch.Tensor] = None, loop: bool = False) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        cell (Tensor): Unit cell vectors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)

    :rtype: :class:`LongTensor` and :class:`Tensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph_pbc

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph_pbc(x, r=1.5, batch=batch, loop=False)
    """
    x_np = x.detach().numpy()
    cell_np = cell.detach().numpy()

    if batch is None:
        batch = torch.zeros(len(x), dtype=torch.long)

    row = []
    col = []
    displacements = []

    for i in batch.unique().tolist():
        ii, = (batch == i).nonzero(as_tuple=True)
        first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
            'ijS',
            (True, True, True),
            cell_np,
            x_np[ii],
            cutoff=r,
            self_interaction=loop,
            use_scaled_positions=False
        )
        if len(second_idex) > 0:
            dx = x[second_idex] - x[first_idex]
            dx = dx + torch.einsum(
                'ni,ij->nj',
                torch.as_tensor(shifts, dtype=x.dtype),
                cell
            )
            row += [ii[torch.as_tensor(first_idex, dtype=torch.long)]]
            col += [ii[torch.as_tensor(second_idex, dtype=torch.long)]]
            displacements += [dx]

    row = torch.cat(row) if row else torch.zeros(0, dtype=torch.long)
    col = torch.cat(col) if col else torch.zeros(0, dtype=torch.long)
    displacements = torch.cat(displacements) if displacements else torch.zeros(0, 3, dtype=x.dtype)

    return torch.stack([row, col], dim=0), displacements
