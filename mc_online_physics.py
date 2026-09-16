from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

from mc_parametric import scaled_spectral_components


class OnlinePhysics(object):
    """Differentiable physical decoder and forward integral for the MC problem.

    The implementation keeps the original physical problem unchanged. The
    smooth background uses fixed Gauss-Legendre quadrature. The Lorentzian term
    uses z=atan((s-m^2)/(m*gamma)), preserving narrow peaks in the integral.
    """

    def __init__(self, args: Any, device: torch.device):
        self.device = device
        self.dtype = torch.float64 if str(args.physics_dtype) == "float64" else torch.float32
        self.model_dtype = torch.float32
        self.noise_level = float(args.noise_level)
        self.data_scale = float(args.data_scale)
        self.shift = float(args.shift)
        self.s_min = float(args.s_min)
        self.s_max = float(args.s_max)
        self.q2_min = float(args.q2_min)
        self.q2_max = float(args.q2_max)
        self.output_points = int(args.output_points)
        self.input_points = int(args.input_points)
        self.integration_points = int(args.integration_points)
        self.resonance_grid_points = int(getattr(args, "resonance_grid_points", 512))
        self.q_block_size = 25

        self.s_output = torch.linspace(
            self.s_min, self.s_max, self.output_points,
            dtype=self.dtype, device=device,
        )
        self.q2_grid = torch.linspace(
            self.q2_min, self.q2_max, self.input_points,
            dtype=self.dtype, device=device,
        )
        self.s_resonance_grid = torch.linspace(
            self.s_min, self.s_max, self.resonance_grid_points,
            dtype=self.model_dtype, device=device,
        )
        nodes, weights = leggauss(self.integration_points)
        self.gl_nodes = torch.as_tensor(nodes, dtype=self.dtype, device=device)
        self.gl_weights = torch.as_tensor(weights, dtype=self.dtype, device=device)

        s_mid = 0.5 * (self.s_max + self.s_min)
        s_half = 0.5 * (self.s_max - self.s_min)
        self.s_fixed = s_mid + s_half * self.gl_nodes
        self.s_fixed_weights = s_half * self.gl_weights

    def components_on_grid(
        self,
        params: torch.Tensor,
        s_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return scaled_spectral_components(
            params.to(device=self.device, dtype=self.model_dtype),
            s_grid.to(device=self.device, dtype=self.model_dtype),
            shift=self.shift,
            data_scale=self.data_scale,
        )

    def components(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.components_on_grid(params, self.s_output)

    def dense_components(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.components_on_grid(params, self.s_resonance_grid)

    def forward_from_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """High-accuracy differentiable g(q^2) from physical parameters."""
        params_work = params.to(device=self.device, dtype=self.dtype)
        if params_work.ndim != 2 or params_work.shape[1] != 5:
            raise ValueError("params must have shape [B, 5]")

        a1 = params_work[:, 0:1]
        a2 = params_work[:, 1:2]
        a3 = params_work[:, 2:3]
        mass = params_work[:, 3:4]
        gamma = params_work[:, 4:5]
        width = (mass * gamma).clamp_min(1e-10)
        center = mass.square()

        background_values = (
            a2 * self.s_fixed.view(1, -1) + a3
        ) / (self.s_fixed.view(1, -1) + self.shift).square()

        z0 = torch.atan((self.s_min - center) / width)
        z1 = torch.atan((self.s_max - center) / width)
        z_mid = 0.5 * (z0 + z1)
        z_half = 0.5 * (z1 - z0)
        z = z_mid + z_half * self.gl_nodes.view(1, -1)
        s_res = center + width * torch.tan(z)
        resonance_coefficient = (
            (a1 / math.pi)
            * z_half
            * self.gl_weights.view(1, -1)
            / (s_res + self.shift).square()
        )

        chunks = []
        for start in range(0, self.input_points, self.q_block_size):
            stop = min(start + self.q_block_size, self.input_points)
            q_chunk = self.q2_grid[start:stop]
            bg_kernel = self.s_fixed_weights.view(-1, 1) / (
                self.s_fixed.view(-1, 1) - q_chunk.view(1, -1)
            )
            background_chunk = background_values @ bg_kernel
            resonance_chunk = torch.sum(
                resonance_coefficient.unsqueeze(2)
                / (s_res.unsqueeze(2) - q_chunk.view(1, 1, -1)),
                dim=1,
            )
            chunks.append(background_chunk + resonance_chunk)
        result = self.data_scale * torch.cat(chunks, dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite forward observations were produced")
        return result.to(self.model_dtype)

    def make_clean_batch(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total, _, _ = self.components(params)
        observation = self.forward_from_parameters(params)
        if not torch.isfinite(total).all():
            raise FloatingPointError("non-finite target curves were produced")
        return total.to(self.model_dtype), observation.to(self.model_dtype)

    def add_noise(
        self,
        clean: torch.Tensor,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(clean.square(), dim=1, keepdim=True).clamp_min(1e-24))
        if generator is None:
            random_values = torch.randn_like(clean)
        else:
            random_values = torch.randn(
                clean.shape,
                generator=generator,
                device=clean.device,
                dtype=clean.dtype,
            )
        return clean + self.noise_level * rms * random_values


def fixed_rms_noise_numpy(clean: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    clean64 = np.asarray(clean, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    rms = np.sqrt(np.mean(clean64 ** 2, axis=1, keepdims=True))
    noise = float(noise_level) * rms * rng.standard_normal(clean64.shape)
    return (clean64 + noise).astype(np.float32)
