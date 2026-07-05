"""
Three-layer feedforward LIF SNN: (23 x 512) -> 64 -> 32 -> 2.

Matches section 2.2 of the paper: shared decay beta=0.9, threshold theta=1.0,
no recurrent connections (snn-mlir compatibility), fast-sigmoid surrogate
gradient with sharpness k=25.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

BETA = 0.9
THETA = 1.0
SURROGATE_K = 25
INPUT_DIM = 23 * 512  # 11,776
HIDDEN1 = 64
HIDDEN2 = 32
OUTPUT_DIM = 2


class SeizureLIFNet(nn.Module):
    """Three LIFLayer nodes, matching the NIR export requirement in 2.2.1."""

    def __init__(self, beta: float = BETA, threshold: float = THETA,
                 k: int = SURROGATE_K):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=k)

        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, init_hidden=False)

        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_DIM, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, init_hidden=False)

    def forward(self, x_flat: torch.Tensor, mem1=None, mem2=None, mem3=None):
        """
        x_flat: (batch, 11776) — a single flattened window, replayed for
        each of the T_steps=512 timesteps by the caller (rate-coded spike
        train is already time-flattened per §2.1.3; here each "timestep"
        of the SNN forward pass corresponds to one flattened window pass,
        matching the compiled `snn_forward_step` semantics in §2.3).

        Membrane potentials (mem1/mem2/mem3) must be threaded through by the
        caller across timesteps -- LIF neurons integrate/leak over time, so
        re-zeroing them every call breaks the temporal dynamics entirely.
        """
        batch = x_flat.shape[0]
        if mem1 is None:
            mem1 = torch.zeros(batch, HIDDEN1, device=x_flat.device)
        if mem2 is None:
            mem2 = torch.zeros(batch, HIDDEN2, device=x_flat.device)
        if mem3 is None:
            mem3 = torch.zeros(batch, OUTPUT_DIM, device=x_flat.device)

        cur1 = self.fc1(x_flat)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem1, mem2, mem3

    def forward_sequence(self, x_flat: torch.Tensor, n_steps: int = 512) -> torch.Tensor:
        """Replay n_steps, carrying membrane potentials across timesteps,
        to get an output spike-count vector per §2.2.3 loss."""
        mem1 = mem2 = mem3 = None
        spikes = []
        for _ in range(n_steps):
            spk, mem1, mem2, mem3 = self.forward(x_flat, mem1, mem2, mem3)
            spikes.append(spk)
        return torch.stack(spikes, dim=0)  # (T, batch, 2)
