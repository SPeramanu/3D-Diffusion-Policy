"""
OCT Staubli dataset for DP3.

Loads zarr replay-buffer with keys: state, action, point_cloud.
The dummy ``img`` key produced by the converter is ignored.

When ``point_cloud`` columns > 3 (e.g. XYZI with intensity), the model
uses as many channels as requested by ``task.shape_meta.obs.point_cloud``.

Stochastic point-cloud augmentation is applied inside ``__getitem__``:
every access produces a fresh random realisation. This mirrors the
``PointCloudNoise`` transform used by the lerobot integration; the
implementation is duplicated here (rather than imported) because DP3
and lerobot live in separate repos.
"""

from typing import Dict, Optional, Sequence, Union
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


# ─────────────────────────────────────────────────────────────────────
# Point-cloud noise transform (mirrors lerobot.datasets.transforms.PointCloudNoise)
# ─────────────────────────────────────────────────────────────────────
def _coerce_sigma_xyz(value: Union[float, Sequence[float]]) -> torch.Tensor:
    """Accept scalar (isotropic) or length-3 sequence (per-axis); return (3,) float32 tensor."""
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)] * 3, dtype=torch.float32)
    seq = list(value)
    if len(seq) == 1:
        return torch.tensor([float(seq[0])] * 3, dtype=torch.float32)
    if len(seq) == 3:
        return torch.tensor([float(x) for x in seq], dtype=torch.float32)
    raise ValueError(
        f"sigma_xyz must be a scalar or length-3 sequence, got length {len(seq)}"
    )


class PointCloudNoise:
    """Stochastic Gaussian noise on (..., N, C>=3) point-cloud tensors.

    Each ``__call__`` independently:
      • draws a Bernoulli(xyz_jitter_rate) — on success, adds N(0, σ_xyz) to XYZ
      • draws a Bernoulli(intensity_jitter_rate) — on success, adds N(0, σ_int)
        to channel 3 (when present)

    Padding rows produced by the converter are exactly (0, 0, 0[, 0]);
    when ``mask_padding=True`` (the default) they are skipped so noise
    does not "wake up" pad rows into fake points.

    σ values are expected in the same coordinate frame the dataset
    actually stores — DP3's zarr holds the converter's normalised XYZ
    (per-frame [-1, 1]) and normalised intensity ([0, 1]).
    """

    def __init__(
        self,
        sigma_xyz: Union[float, Sequence[float]] = 0.0,
        sigma_intensity: float = 0.0,
        xyz_jitter_rate: float = 1.0,
        intensity_jitter_rate: float = 1.0,
        mask_padding: bool = True,
    ) -> None:
        self._sigma_xyz = _coerce_sigma_xyz(sigma_xyz)
        if torch.any(self._sigma_xyz < 0):
            raise ValueError(
                f"sigma_xyz must be >= 0 on all axes, got {self._sigma_xyz.tolist()}"
            )
        if sigma_intensity < 0:
            raise ValueError(f"sigma_intensity must be >= 0, got {sigma_intensity}")
        if not 0.0 <= xyz_jitter_rate <= 1.0:
            raise ValueError(f"xyz_jitter_rate must be in [0, 1], got {xyz_jitter_rate}")
        if not 0.0 <= intensity_jitter_rate <= 1.0:
            raise ValueError(
                f"intensity_jitter_rate must be in [0, 1], got {intensity_jitter_rate}"
            )

        self._sigma_intensity = float(sigma_intensity)
        self._xyz_rate = float(xyz_jitter_rate)
        self._intensity_rate = float(intensity_jitter_rate)
        self._mask_padding = bool(mask_padding)

        # rate=0 OR σ=0 collapses a branch to a no-op.
        self._xyz_enabled = bool(torch.any(self._sigma_xyz > 0).item()) and self._xyz_rate > 0
        self._intensity_enabled = self._sigma_intensity > 0 and self._intensity_rate > 0

    def __bool__(self) -> bool:
        return self._xyz_enabled or self._intensity_enabled

    @staticmethod
    def _bernoulli(p: float) -> bool:
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return bool(torch.rand(1).item() < p)

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        if not self._xyz_enabled and not self._intensity_enabled:
            return pts
        if pts.ndim < 2 or pts.shape[-1] < 3:
            raise ValueError(
                f"PointCloudNoise expected (..., N, C>=3), got shape {tuple(pts.shape)}"
            )

        apply_xyz = self._xyz_enabled and self._bernoulli(self._xyz_rate)
        apply_intensity = (
            self._intensity_enabled
            and pts.shape[-1] >= 4
            and self._bernoulli(self._intensity_rate)
        )
        if not apply_xyz and not apply_intensity:
            return pts

        out = pts.to(dtype=torch.float32).clone()

        if self._mask_padding:
            xyz = out[..., :3]
            non_pad = (xyz != 0).any(dim=-1)
        else:
            non_pad = None

        if apply_xyz:
            sigma = self._sigma_xyz.to(device=out.device, dtype=out.dtype)
            noise = torch.randn_like(out[..., :3]) * sigma
            if non_pad is not None:
                noise = noise * non_pad.unsqueeze(-1).to(noise.dtype)
            out[..., :3] = out[..., :3] + noise

        if apply_intensity:
            noise_i = torch.randn_like(out[..., 3]) * self._sigma_intensity
            if non_pad is not None:
                noise_i = noise_i * non_pad.to(noise_i.dtype)
            out[..., 3] = out[..., 3] + noise_i

        return out

    def __repr__(self) -> str:
        return (
            f"PointCloudNoise(sigma_xyz={self._sigma_xyz.tolist()}, "
            f"sigma_intensity={self._sigma_intensity}, "
            f"xyz_jitter_rate={self._xyz_rate}, "
            f"intensity_jitter_rate={self._intensity_rate}, "
            f"mask_padding={self._mask_padding})"
        )


class OCTDataset(BaseDataset):
    """Dataset for OCT point-cloud + joint-state trajectories."""

    # Only the keys we actually need from the zarr.
    REQUIRED_KEYS = ['state', 'action', 'point_cloud']

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        task_name: Optional[str] = None,
        # ── Stochastic point-cloud augmentation (training-time, per-getitem) ──
        # All defaults below are inert. The transform short-circuits to identity
        # when either ``augment_pointcloud=False`` OR every σ is 0 OR every
        # rate is 0. Override via Hydra: `task.dataset.augment_pointcloud=true`
        # `task.dataset.sigma_xyz=[0.012,0.0013,0.0018]` etc.
        augment_pointcloud: bool = False,
        sigma_xyz: Union[float, Sequence[float]] = 0.0,
        sigma_intensity: float = 0.0,
        xyz_jitter_rate: float = 1.0,
        intensity_jitter_rate: float = 1.0,
        mask_padding: bool = True,
        augment_val: bool = False,
    ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=self.REQUIRED_KEYS)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Build the augmentation transform once. Collapses to a no-op when
        # both σ values (or both rates) are zero, so the per-getitem call
        # short-circuits without allocating.
        self._pc_noise = PointCloudNoise(
            sigma_xyz=sigma_xyz,
            sigma_intensity=sigma_intensity,
            xyz_jitter_rate=xyz_jitter_rate,
            intensity_jitter_rate=intensity_jitter_rate,
            mask_padding=mask_padding,
        )
        # Master toggle. When False the transform is bypassed regardless of
        # any σ / rate values — useful for ablation experiments without
        # having to clear every config field.
        self._augment_enabled = bool(augment_pointcloud)
        # ``augment_val=False`` (default) disables augmentation on the
        # validation split — the val dataset is created via copy.copy below
        # and we flip this flag there.
        self._augment_val = bool(augment_val)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        # Disable augmentation on validation unless explicitly requested
        # (augment_val=True). If the master toggle is off, val stays off
        # regardless of augment_val. copy.copy shares the same _pc_noise
        # instance so we gate via the flag rather than reconstructing.
        val_set._augment_enabled = self._augment_enabled and self._augment_val
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,  # (T, N_points, C) where C >= 3
                'agent_pos': agent_pos,       # (T, D_state)
            },
            'action': sample['action'].astype(np.float32),  # (T, D_action)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        # Stochastic per-call augmentation. ``_pc_noise`` returns the same
        # object if its config is inert OR if both Bernoulli gates miss
        # this sample, so the no-op branch is cheap.
        if self._augment_enabled and bool(self._pc_noise):
            torch_data['obs']['point_cloud'] = self._pc_noise(
                torch_data['obs']['point_cloud']
            )
        return torch_data
