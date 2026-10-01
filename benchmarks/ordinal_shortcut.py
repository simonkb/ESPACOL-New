"""Deterministic controlled ordinal shortcuts with exact pixel counterfactuals.

The benchmark encodes the four APTOS ordinal targets ``1[y > k]`` using four
independently corrupted codewords.  It deliberately keeps code generation
outside the clinical preprocessing transform: a clean, canonical image can
therefore be replayed with any marker bit while every other pixel remains
identical.  Marker locations, appearances, corruption draws, and image
permutations are all derived from content-independent cryptographic seeds.

``OrdinalShortcutDataset.__getitem__`` preserves ORIGIN's ordinary four-field
training contract.  Auditors call :meth:`OrdinalShortcutDataset.render` to get
the same image together with hidden procedural supports and provenance.  This
separation prevents a learner from receiving the localization answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SHORTCUT_FAMILIES = ("localized", "border", "diffuse")
SHORTCUT_CONDITIONS = (
    "aligned",
    "neutral",
    "inverted",
    "missing",
    "boundary_swapped",
    "conflicting",
    "clean",
    "cue_only",
)
POSITION_DOMAINS = ("seen", "unseen")
APPEARANCE_DOMAINS = ("seen", "unseen")
ORDINAL_SHORTCUT_VERSION = "origin-ordinal-shortcut-v1"


def _stable_u64(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"orig-short-v1")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little", signed=False)


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(_stable_u64(*parts))


def _as_bool_tuple(values: Sequence[bool | int], length: int) -> tuple[bool, ...]:
    result = tuple(bool(value) for value in values)
    if len(result) != length:
        raise ValueError(f"expected {length} marker values, got {len(result)}")
    return result


@contextmanager
def _isolated_rng(seed: int) -> Iterator[None]:
    """Seed stochastic base transforms without changing caller RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) % (2**63 - 1))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass(frozen=True)
class OrdinalShortcutProtocol:
    """Frozen, serializable construction policy for one benchmark arm."""

    seed: int = 271828
    num_boundaries: int = 4
    aligned_true_positive: float = 0.90
    aligned_false_positive: float = 0.10
    strength: float = 0.30
    marker_radius_fraction: float = 0.040
    min_marker_radius: int = 2
    deterministic_base_transform: bool = True
    version: str = ORDINAL_SHORTCUT_VERSION

    def __post_init__(self) -> None:
        if self.version != ORDINAL_SHORTCUT_VERSION:
            raise ValueError(f"unsupported shortcut protocol {self.version!r}")
        if self.num_boundaries != 4:
            raise ValueError("the preregistered APTOS benchmark has four boundaries")
        probabilities = (self.aligned_true_positive, self.aligned_false_positive)
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("marker probabilities must lie in [0, 1]")
        if self.aligned_true_positive <= self.aligned_false_positive:
            raise ValueError("aligned markers must be positively correlated with targets")
        if not math.isfinite(self.strength) or self.strength <= 0.0:
            raise ValueError("strength must be finite and positive")
        if not 0.0 < self.marker_radius_fraction < 0.25:
            raise ValueError("marker_radius_fraction must lie in (0, .25)")
        if self.min_marker_radius < 1:
            raise ValueError("min_marker_radius must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": int(self.seed),
            "num_boundaries": int(self.num_boundaries),
            "aligned_true_positive": float(self.aligned_true_positive),
            "aligned_false_positive": float(self.aligned_false_positive),
            "strength": float(self.strength),
            "marker_radius_fraction": float(self.marker_radius_fraction),
            "min_marker_radius": int(self.min_marker_radius),
            "deterministic_base_transform": bool(self.deterministic_base_transform),
        }

    @property
    def signature(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ShortcutMetadata:
    """Hidden ground truth for a rendered shortcut sample.

    ``support_masks`` exist even for a missing or clean rendering and define
    where a cue *would* be placed. ``rendered_masks`` are the actually changed
    supports.  Both tensors have shape ``(boundary, H, W)``.
    """

    sample_index: int
    source_index: int
    label: int
    target_bits: tuple[bool, ...]
    marker_bits: tuple[bool, ...]
    marker_present: tuple[bool, ...]
    source_boundaries: tuple[int, ...]
    family: str
    condition: str
    position_domain: str
    appearance_domain: str
    sample_seed: int
    support_masks: torch.Tensor
    rendered_masks: torch.Tensor
    centers_yx: tuple[tuple[float, float], ...]
    appearance_ids: tuple[int, ...]
    localization_applicable: bool
    support_kind: str
    protocol_signature: str

    def __post_init__(self) -> None:
        masks = self.support_masks
        rendered = self.rendered_masks
        if masks.dtype != torch.bool or rendered.dtype != torch.bool:
            raise TypeError("shortcut masks must be boolean")
        if masks.ndim != 3 or masks.shape != rendered.shape:
            raise ValueError("shortcut masks must have matching (B,H,W) shapes")
        boundaries = masks.shape[0]
        for values in (
            self.target_bits,
            self.marker_bits,
            self.marker_present,
            self.source_boundaries,
            self.centers_yx,
            self.appearance_ids,
        ):
            if len(values) != boundaries:
                raise ValueError("metadata boundary fields do not match mask count")
        expected_rendered = masks & torch.tensor(
            self.marker_present, dtype=torch.bool, device=masks.device
        )[:, None, None]
        if not torch.equal(rendered, expected_rendered):
            raise ValueError("rendered masks must equal support masks at present cues")

    @property
    def union_support(self) -> torch.Tensor:
        return self.support_masks.any(dim=0)

    @property
    def union_rendered(self) -> torch.Tensor:
        return self.rendered_masks.any(dim=0)

    def to_record(self, *, include_masks: bool = False) -> dict[str, Any]:
        mask_bytes = self.support_masks.detach().cpu().numpy().tobytes()
        record: dict[str, Any] = {
            "sample_index": self.sample_index,
            "source_index": self.source_index,
            "label": self.label,
            "target_bits": list(self.target_bits),
            "marker_bits": list(self.marker_bits),
            "marker_present": list(self.marker_present),
            "source_boundaries": list(self.source_boundaries),
            "family": self.family,
            "condition": self.condition,
            "position_domain": self.position_domain,
            "appearance_domain": self.appearance_domain,
            "sample_seed": self.sample_seed,
            "centers_yx": [list(value) for value in self.centers_yx],
            "appearance_ids": list(self.appearance_ids),
            "localization_applicable": self.localization_applicable,
            "support_kind": self.support_kind,
            "protocol_signature": self.protocol_signature,
            "mask_shape": list(self.support_masks.shape),
            "mask_pixels": self.support_masks.sum(dim=(-2, -1)).tolist(),
            "rendered_pixels": self.rendered_masks.sum(dim=(-2, -1)).tolist(),
            "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        }
        if include_masks:
            record["support_masks"] = self.support_masks.tolist()
            record["rendered_masks"] = self.rendered_masks.tolist()
        return record


@dataclass(frozen=True)
class ShortcutSample:
    image: torch.Tensor
    clean_image: torch.Tensor
    valid_mask: torch.Tensor
    label: torch.Tensor
    stable_index: int
    metadata: ShortcutMetadata

    def training_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.image, self.valid_mask, self.label, self.stable_index


def _labels_from_base(dataset: Dataset) -> list[int]:
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None and len(values) == len(dataset):
            return [int(value) for value in values]
    items = getattr(dataset, "items", None)
    if items is not None and len(items) == len(dataset):
        return [int(item[1]) for item in items]
    labels: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise TypeError("base dataset must return image, valid mask, label, index")
        labels.append(int(torch.as_tensor(sample[2]).item()))
    return labels


def _derangement(length: int, seed: int) -> tuple[int, ...]:
    if length < 2:
        raise ValueError("cue-only image permutation requires at least two samples")
    order = np.arange(length)
    generator = _rng(seed, "image-derangement", length)
    for _ in range(100):
        candidate = generator.permutation(length)
        if np.all(candidate != order):
            return tuple(int(value) for value in candidate)
    # A deterministic cyclic fallback is always a derangement for n > 1.
    shift = 1 + int(_stable_u64(seed, "fallback-shift") % (length - 1))
    return tuple(int(value) for value in np.roll(order, shift))


def _canonical_valid_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask).bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("base valid mask must have shape (H,W) or (1,H,W)")
    if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
        raise ValueError("base image and valid mask spatial shapes differ")
    return mask


def _disk_mask(
    height: int, width: int, center_y: float, center_x: float, radius: float
) -> torch.Tensor:
    yy = torch.arange(height, dtype=torch.float32)[:, None]
    xx = torch.arange(width, dtype=torch.float32)[None, :]
    return (yy - center_y).square() + (xx - center_x).square() <= radius**2


def _candidate_centers(
    height: int,
    width: int,
    valid: torch.Tensor,
    *,
    family: str,
    domain: str,
    radius: int,
    seed: int,
    boundaries: int,
) -> tuple[tuple[float, float], ...]:
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    scale_y = max(1.0, height / 2.0)
    scale_x = max(1.0, width / 2.0)

    if family == "localized":
        offset = 0.0 if domain == "seen" else math.pi / 8.0
        radii = (0.25, 0.48) if domain == "seen" else (0.36, 0.60)
        angles = [offset + index * math.pi / 4.0 for index in range(8)]
        normalized = [
            (radial * math.sin(angle), radial * math.cos(angle))
            for radial in radii
            for angle in angles
        ]
    elif family == "border":
        offset = 0.0 if domain == "seen" else math.pi / 8.0
        normalized = [
            (0.82 * math.sin(offset + index * math.pi / 4.0),
             0.82 * math.cos(offset + index * math.pi / 4.0))
            for index in range(8)
        ]
    else:
        return tuple((center_y, center_x) for _ in range(boundaries))

    candidates: list[tuple[float, float]] = []
    for norm_y, norm_x in normalized:
        y = center_y + norm_y * scale_y
        x = center_x + norm_x * scale_x
        support = _disk_mask(height, width, y, x, radius)
        valid_fraction = float((support & valid).sum()) / max(1, int(support.sum()))
        if valid_fraction >= (0.85 if family == "localized" else 0.45):
            candidates.append((y, x))
    if len(candidates) < boundaries:
        # Tiny synthetic test images may not resolve the full annular lattice.
        inset_y = max(radius + 1, int(round(height * 0.25)))
        inset_x = max(radius + 1, int(round(width * 0.25)))
        fallback = [
            (float(inset_y), float(inset_x)),
            (float(inset_y), float(width - 1 - inset_x)),
            (float(height - 1 - inset_y), float(inset_x)),
            (float(height - 1 - inset_y), float(width - 1 - inset_x)),
        ]
        candidates.extend(fallback)

    order = _rng(seed, family, domain, "position-order").permutation(len(candidates))
    selected: list[tuple[float, float]] = []
    minimum_distance = max(2.1 * radius, 2.0)
    for candidate_index in order:
        candidate = candidates[int(candidate_index)]
        if all(math.dist(candidate, previous) >= minimum_distance for previous in selected):
            selected.append(candidate)
        if len(selected) == boundaries:
            break
    if len(selected) != boundaries:
        raise RuntimeError("could not place four nonoverlapping shortcut codes")
    # Boundary-to-position identity is independently permuted for every image.
    assignment = _rng(seed, family, domain, "boundary-position").permutation(boundaries)
    return tuple(selected[int(index)] for index in assignment)


def _support_masks(
    valid: torch.Tensor,
    *,
    family: str,
    position_domain: str,
    radius: int,
    seed: int,
    boundaries: int,
) -> tuple[torch.Tensor, tuple[tuple[float, float], ...]]:
    height, width = valid.shape
    centers = _candidate_centers(
        height,
        width,
        valid,
        family=family,
        domain=position_domain,
        radius=radius,
        seed=seed,
        boundaries=boundaries,
    )
    if family == "diffuse":
        masks = valid.unsqueeze(0).expand(boundaries, -1, -1).clone()
        return masks, centers

    masks: list[torch.Tensor] = []
    yy = torch.arange(height, dtype=torch.float32)[:, None]
    xx = torch.arange(width, dtype=torch.float32)[None, :]
    for center_y, center_x in centers:
        distance = ((yy - center_y) / max(radius, 1)).square() + (
            (xx - center_x) / max(radius, 1)
        ).square()
        if family == "localized":
            # A filled core plus punctate annulus reads as lesion-like without
            # changing the active/inactive area or mean colour.
            support = distance <= 1.0
        else:
            # Acquisition artifacts live in a short retinal-edge arc rather
            # than an artificial square in the image centre.
            support = (distance <= 1.8) & (distance >= 0.12)
        support &= valid
        masks.append(support)
    stacked = torch.stack(masks)
    if family == "localized":
        overlap = stacked.to(torch.int16).sum(dim=0)
        if bool((overlap > 1).any()):
            raise RuntimeError("localized shortcut supports overlap")
    return stacked, centers


def _balanced_sign(mask: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """A {-1,0,+1} pattern with the same multiset for every codeword."""

    positions = mask.flatten().nonzero(as_tuple=False).flatten()
    result = torch.zeros(mask.numel(), dtype=torch.float32)
    if positions.numel() < 2:
        return result.reshape_as(mask)
    ordered = positions[torch.argsort(score.flatten()[positions], stable=True)]
    pairs = int(ordered.numel() // 2)
    result[ordered[:pairs]] = -1.0
    result[ordered[-pairs:]] = 1.0
    return result.reshape_as(mask)


_BOUNDARY_COLOURS = torch.tensor(
    [
        [1.00, -0.25, -0.25],
        [0.75, 0.75, -0.35],
        [-0.30, 0.85, 0.85],
        [0.85, -0.30, 0.85],
    ],
    dtype=torch.float32,
)


def _code_perturbation(
    mask: torch.Tensor,
    *,
    center: tuple[float, float],
    family: str,
    boundary: int,
    bit: bool,
    appearance_domain: str,
    appearance_id: int,
    strength: float,
) -> torch.Tensor:
    height, width = mask.shape
    yy = torch.arange(height, dtype=torch.float32)[:, None]
    xx = torch.arange(width, dtype=torch.float32)[None, :]
    cy, cx = center
    theta_offset = (appearance_id % 7) * math.pi / 28.0
    if appearance_domain == "unseen":
        theta_offset += math.pi / 9.0
    theta = theta_offset + boundary * math.pi / 11.0
    along = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
    across = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)

    if family == "localized":
        frequency = (2.0 if appearance_domain == "seen" else 3.5) + appearance_id % 2
        chosen = along if bit else across
        score = torch.sin(chosen * math.pi / max(2.0, min(height, width) / frequency))
        score += 0.05 * torch.cos((along + across) * (boundary + 1))
        scalar = _balanced_sign(mask, score)
    elif family == "border":
        frequency = (1.5 if appearance_domain == "seen" else 2.75) + appearance_id % 3
        chosen = along + 0.3 * across if bit else across - 0.3 * along
        score = torch.cos(chosen * frequency)
        scalar = _balanced_sign(mask, score)
    else:
        # Smooth zero-mean illumination/colour fields deliberately have global
        # support.  They are not entitled to a focal-lesion interpretation.
        norm_x = (xx - (width - 1) / 2.0) / max(width - 1, 1)
        norm_y = (yy - (height - 1) / 2.0) / max(height - 1, 1)
        harmonic = (boundary + 1) * (1.0 if appearance_domain == "seen" else 1.5)
        phase = (appearance_id % 9) * math.pi / 9.0
        if bit:
            scalar = torch.sin(2.0 * math.pi * harmonic * norm_x + phase)
        else:
            scalar = torch.sin(2.0 * math.pi * harmonic * norm_y + phase)
        scalar = scalar * mask
        values = scalar[mask]
        if values.numel():
            scalar = scalar.clone()
            scalar[mask] -= values.mean()
            rms = scalar[mask].square().mean().sqrt().clamp_min(1e-6)
            scalar[mask] /= rms

    colour = _BOUNDARY_COLOURS[boundary]
    jitter = 0.85 + 0.30 * ((_stable_u64(appearance_id, boundary, family) % 10001) / 10000.0)
    return float(strength * jitter) * colour[:, None, None] * scalar[None]


def _sample_marker_state(
    *,
    label: int,
    condition: str,
    protocol: OrdinalShortcutProtocol,
    sample_seed: int,
    override_bits: Sequence[bool | int] | None,
    override_present: Sequence[bool | int] | None,
) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...], tuple[int, ...]]:
    boundaries = protocol.num_boundaries
    target = tuple(label > boundary for boundary in range(boundaries))
    source_boundaries = tuple(range(boundaries))

    if condition == "boundary_swapped":
        # A derangement makes boundary identity, rather than mere cue presence,
        # necessary. The mapping is fixed before outcomes are inspected.
        source_boundaries = (1, 0, 3, 2)
    source_target = tuple(target[index] for index in source_boundaries)

    bits: list[bool] = []
    present = [True] * boundaries
    for boundary in range(boundaries):
        draw = float(_rng(sample_seed, "marker-noise", boundary).random())
        if condition in {"aligned", "missing", "boundary_swapped"}:
            probability = (
                protocol.aligned_true_positive
                if source_target[boundary]
                else protocol.aligned_false_positive
            )
            bits.append(draw < probability)
        elif condition == "neutral":
            bits.append(draw < 0.5)
        elif condition == "inverted":
            probability = (
                protocol.aligned_false_positive
                if source_target[boundary]
                else protocol.aligned_true_positive
            )
            bits.append(draw < probability)
        elif condition == "conflicting":
            # Alternating non-nested codewords cannot represent any valid
            # ordinal grade and therefore make the conflict explicit.
            parity = int(_stable_u64(sample_seed, "conflict") & 1)
            bits.append(bool((boundary + parity) % 2))
        elif condition == "cue_only":
            bits.append(target[boundary])
        elif condition == "clean":
            bits.append(False)
            present[boundary] = False
        else:
            raise ValueError(f"unsupported shortcut condition {condition!r}")

    if condition == "missing":
        present = [
            bool(_rng(sample_seed, "missing", boundary).random() >= 0.5)
            for boundary in range(boundaries)
        ]
        if all(present):
            present[int(_stable_u64(sample_seed, "force-missing") % boundaries)] = False

    if override_bits is not None:
        bits = list(_as_bool_tuple(override_bits, boundaries))
    if override_present is not None:
        present = list(_as_bool_tuple(override_present, boundaries))
    return target, tuple(bits), tuple(present), source_boundaries


class OrdinalShortcutDataset(Dataset):
    """Wrap a clean ORIGIN dataset with a deterministic shortcut arm."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        family: str,
        condition: str = "aligned",
        position_domain: str = "seen",
        appearance_domain: str = "seen",
        protocol: OrdinalShortcutProtocol | None = None,
        permute_images: bool | None = None,
    ) -> None:
        family = str(family).lower()
        condition = str(condition).lower()
        position_domain = str(position_domain).lower()
        appearance_domain = str(appearance_domain).lower()
        if family not in SHORTCUT_FAMILIES:
            raise ValueError(f"family must be one of {SHORTCUT_FAMILIES}")
        if condition not in SHORTCUT_CONDITIONS:
            raise ValueError(f"condition must be one of {SHORTCUT_CONDITIONS}")
        if position_domain not in POSITION_DOMAINS:
            raise ValueError(f"position_domain must be one of {POSITION_DOMAINS}")
        if appearance_domain not in APPEARANCE_DOMAINS:
            raise ValueError(f"appearance_domain must be one of {APPEARANCE_DOMAINS}")
        if len(base_dataset) < 1:
            raise ValueError("base dataset is empty")

        self.base_dataset = base_dataset
        self.family = family
        self.condition = condition
        self.position_domain = position_domain
        self.appearance_domain = appearance_domain
        self.protocol = protocol or OrdinalShortcutProtocol()
        self.labels = _labels_from_base(base_dataset)
        self.targets = self.labels
        self.items = getattr(base_dataset, "items", None)
        should_permute = condition == "cue_only" if permute_images is None else bool(permute_images)
        if condition == "cue_only" and not should_permute:
            raise ValueError("cue_only must permute clean image-label assignments")
        self.image_permutation = (
            _derangement(len(self), self.protocol.seed) if should_permute else tuple(range(len(self)))
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _base_sample(self, source_index: int, sample_index: int):
        seed = _stable_u64(
            self.protocol.seed,
            "base-transform",
            sample_index,
            source_index,
            self.position_domain,
            self.appearance_domain,
        )
        if self.protocol.deterministic_base_transform:
            with _isolated_rng(seed):
                return self.base_dataset[source_index]
        return self.base_dataset[source_index]

    def render(
        self,
        index: int,
        *,
        marker_bits: Sequence[bool | int] | None = None,
        marker_present: Sequence[bool | int] | None = None,
    ) -> ShortcutSample:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        index = int(index)
        source_index = int(self.image_permutation[index])
        raw = self._base_sample(source_index, index)
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise TypeError("base dataset must return image, valid mask, label, index")
        clean = torch.as_tensor(raw[0]).detach().clone()
        if clean.ndim != 3 or clean.shape[0] != 3 or not clean.is_floating_point():
            raise ValueError("base image must be a floating (3,H,W) tensor")
        valid_mask = _canonical_valid_mask(torch.as_tensor(raw[1]), clean)
        label = int(self.labels[index])
        if label < 0 or label > self.protocol.num_boundaries:
            raise ValueError(
                f"label {label} cannot be represented by {self.protocol.num_boundaries} boundaries"
            )
        sample_seed = _stable_u64(
            self.protocol.seed,
            "sample",
            index,
            self.family,
            self.position_domain,
            self.appearance_domain,
        )
        target, bits, present, source_boundaries = _sample_marker_state(
            label=label,
            condition=self.condition,
            protocol=self.protocol,
            sample_seed=sample_seed,
            override_bits=marker_bits,
            override_present=marker_present,
        )
        height, width = clean.shape[-2:]
        radius = max(
            self.protocol.min_marker_radius,
            int(round(min(height, width) * self.protocol.marker_radius_fraction)),
        )
        supports, centers = _support_masks(
            valid_mask[0],
            family=self.family,
            position_domain=self.position_domain,
            radius=radius,
            seed=sample_seed,
            boundaries=self.protocol.num_boundaries,
        )
        appearances = tuple(
            int(_stable_u64(sample_seed, "appearance", boundary) % 1_000_003)
            for boundary in range(self.protocol.num_boundaries)
        )

        image = clean.clone()
        for boundary, is_present in enumerate(present):
            if not is_present:
                continue
            image += _code_perturbation(
                supports[boundary],
                center=centers[boundary],
                family=self.family,
                boundary=boundary,
                bit=bits[boundary],
                appearance_domain=self.appearance_domain,
                appearance_id=appearances[boundary],
                strength=self.protocol.strength,
            ).to(device=image.device, dtype=image.dtype)

        rendered = supports & torch.tensor(present, dtype=torch.bool)[:, None, None]
        metadata = ShortcutMetadata(
            sample_index=index,
            source_index=source_index,
            label=label,
            target_bits=target,
            marker_bits=bits,
            marker_present=present,
            source_boundaries=source_boundaries,
            family=self.family,
            condition=self.condition,
            position_domain=self.position_domain,
            appearance_domain=self.appearance_domain,
            sample_seed=sample_seed,
            support_masks=supports,
            rendered_masks=rendered,
            centers_yx=centers,
            appearance_ids=appearances,
            localization_applicable=self.family != "diffuse",
            support_kind="global_diffuse" if self.family == "diffuse" else (
                "retinal_edge" if self.family == "border" else "focal_retinal"
            ),
            protocol_signature=self.protocol.signature,
        )
        return ShortcutSample(
            image=image,
            clean_image=clean,
            valid_mask=valid_mask,
            label=torch.tensor(label, dtype=torch.long),
            stable_index=index,
            metadata=metadata,
        )

    def __getitem__(self, index: int):
        return self.render(index).training_tuple()

    def counterfactual_pair(
        self, index: int, boundary: int
    ) -> tuple[ShortcutSample, ShortcutSample]:
        if not 0 <= int(boundary) < self.protocol.num_boundaries:
            raise ValueError("boundary is outside the shortcut ledger")
        reference = self.render(index)
        lower = list(reference.metadata.marker_bits)
        upper = list(reference.metadata.marker_bits)
        present = list(reference.metadata.marker_present)
        lower[int(boundary)] = False
        upper[int(boundary)] = True
        present[int(boundary)] = True
        return (
            self.render(index, marker_bits=lower, marker_present=present),
            self.render(index, marker_bits=upper, marker_present=present),
        )


class FactorialShortcutDataset(Dataset):
    """All 16 marker states for every clean image, with exact pairing."""

    def __init__(self, dataset: OrdinalShortcutDataset) -> None:
        if dataset.protocol.num_boundaries != 4:
            raise ValueError("the factorial benchmark requires four markers")
        self.dataset = dataset
        self.factor = 2 ** dataset.protocol.num_boundaries
        self.labels = [label for label in dataset.labels for _ in range(self.factor)]
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.dataset) * self.factor

    def decode_index(self, index: int) -> tuple[int, tuple[bool, ...]]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        base_index, combination = divmod(int(index), self.factor)
        bits = tuple(
            bool((combination >> boundary) & 1)
            for boundary in range(self.dataset.protocol.num_boundaries)
        )
        return base_index, bits

    def render(self, index: int) -> ShortcutSample:
        base_index, bits = self.decode_index(index)
        sample = self.dataset.render(
            base_index,
            marker_bits=bits,
            marker_present=(True,) * self.dataset.protocol.num_boundaries,
        )
        return replace(sample, stable_index=int(index))

    def __getitem__(self, index: int):
        return self.render(index).training_tuple()


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + int(worker_id))
    np.random.seed((seed + int(worker_id)) % (2**32))


def make_shortcut_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build deterministic loaders without leaking shortcut metadata."""

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator().manual_seed(int(seed))
    common: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": False,
        "worker_init_fn": _worker_seed,
    }
    if num_workers:
        common["prefetch_factor"] = 2
    train = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation = DataLoader(validation_dataset, shuffle=False, **common)
    test = DataLoader(test_dataset, shuffle=False, **common)
    return train, validation, test


__all__ = [
    "APPEARANCE_DOMAINS",
    "ORDINAL_SHORTCUT_VERSION",
    "POSITION_DOMAINS",
    "SHORTCUT_CONDITIONS",
    "SHORTCUT_FAMILIES",
    "FactorialShortcutDataset",
    "OrdinalShortcutDataset",
    "OrdinalShortcutProtocol",
    "ShortcutMetadata",
    "ShortcutSample",
    "make_shortcut_loaders",
]
