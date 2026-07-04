"""
EEG -> spike-tensor preprocessing.

Implements the pipeline described in section 2.1 of the paper:
  - 23-channel 10-20 subset, 256 Hz
  - non-overlapping... actually 2s windows with 1s step (WIN=512 timesteps)
  - per-channel rate coding: spike whenever amplitude > mu + 2*sigma
  - random undersampling to a 5:1 non-seizure:seizure ratio

Two data sources are supported:
  1. Real CHB-MIT recordings (requires `mne` + local PhysioNet download)
  2. A synthetic generator (default), so the pipeline runs offline.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

WINDOW_SEC = 2
STEP_SEC = 1
FS_HZ = 256
WIN = WINDOW_SEC * FS_HZ  # 512
N_CHANNELS = 23
BANDPASS_HZ = (0.5, 40.0)
UNDERSAMPLE_RATIO = 5  # non-seizure : seizure


@dataclasses.dataclass
class SpikeDataset:
    spikes: np.ndarray   # (N, 23, 512) binary
    labels: np.ndarray   # (N,) 0 = non-seizure, 1 = seizure


def rate_code(amplitude: np.ndarray) -> np.ndarray:
    """Per-channel, per-window rate coding: spike if amplitude > mu + 2*sigma."""
    mu = amplitude.mean(axis=-1, keepdims=True)
    sigma = amplitude.std(axis=-1, keepdims=True) + 1e-8
    threshold = mu + 2.0 * sigma
    return (amplitude > threshold).astype(np.float32)


def _undersample(spikes: np.ndarray, labels: np.ndarray, seed: int = 0) -> SpikeDataset:
    rng = np.random.default_rng(seed)
    seizure_idx = np.where(labels == 1)[0]
    non_idx = np.where(labels == 0)[0]
    cap = min(len(non_idx), UNDERSAMPLE_RATIO * len(seizure_idx))
    keep_non = rng.choice(non_idx, size=cap, replace=False)
    keep = np.concatenate([seizure_idx, keep_non])
    rng.shuffle(keep)
    return SpikeDataset(spikes=spikes[keep], labels=labels[keep])


def load_synthetic(n_seizure: int = 604, n_nonseizure_pool: int = 6000,
                    seed: int = 0) -> SpikeDataset:
    """
    Synthetic stand-in for CHB-MIT chb01+chb02, matching Table 1 window
    counts by default (604 seizure windows before the 5:1 cap).

    Seizure windows get a higher-amplitude, higher-variance signal so the
    rate-coding step produces a learnable spike-count difference, without
    claiming clinical realism.
    """
    rng = np.random.default_rng(seed)

    non_amp = rng.normal(0.0, 1.0, size=(n_nonseizure_pool, N_CHANNELS, WIN))
    sei_amp = rng.normal(0.0, 1.0, size=(n_seizure, N_CHANNELS, WIN))
    # inject bursty, higher-amplitude activity on a subset of channels
    burst_channels = rng.choice(N_CHANNELS, size=N_CHANNELS // 2, replace=False)
    sei_amp[:, burst_channels, :] += rng.normal(2.5, 0.7, size=(n_seizure, len(burst_channels), WIN))

    amp = np.concatenate([non_amp, sei_amp], axis=0)
    labels = np.concatenate([
        np.zeros(n_nonseizure_pool, dtype=np.int64),
        np.ones(n_seizure, dtype=np.int64),
    ])

    spikes = rate_code(amp)
    return _undersample(spikes, labels, seed=seed)


def load_chbmit(chbmit_dir: str, patients: list[str], seed: int = 0) -> SpikeDataset:
    """
    Load and window real CHB-MIT recordings via MNE. Requires local
    PhysioNet EDF files + the accompanying seizure summary annotations
    (`*-summary.txt`) under `chbmit_dir/<patient>/`.
    """
    import mne  # local import: optional heavy dependency

    all_amp, all_labels = [], []
    for patient in patients:
        pdir = Path(chbmit_dir) / patient
        edf_files = sorted(pdir.glob("*.edf"))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found under {pdir}")

        for edf_path in edf_files:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick(raw.ch_names[:N_CHANNELS])
            raw.filter(*BANDPASS_HZ, verbose=False)
            raw.resample(FS_HZ, verbose=False)
            data = raw.get_data()  # (channels, samples)

            seizure_intervals = _parse_seizure_summary(pdir, edf_path.name)

            n_samples = data.shape[1]
            step = STEP_SEC * FS_HZ
            for start in range(0, n_samples - WIN + 1, step):
                end = start + WIN
                window = data[:N_CHANNELS, start:end]
                if window.shape[1] < WIN:
                    continue
                t0, t1 = start / FS_HZ, end / FS_HZ
                label = int(any(a <= t1 and b >= t0 for a, b in seizure_intervals))
                all_amp.append(window)
                all_labels.append(label)

    amp = np.stack(all_amp, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    spikes = rate_code(amp)
    return _undersample(spikes, labels, seed=seed)


def _parse_seizure_summary(patient_dir: Path, edf_name: str) -> list[tuple[float, float]]:
    """Parse '<patient>-summary.txt' for seizure start/end times (seconds)."""
    summary_files = list(patient_dir.glob("*-summary.txt"))
    if not summary_files:
        return []
    text = summary_files[0].read_text(errors="ignore")
    intervals: list[tuple[float, float]] = []
    block = []
    capture = False
    for line in text.splitlines():
        if line.startswith("File Name:"):
            capture = edf_name in line
            block = []
        if capture and "Seizure" in line and "Start Time" in line:
            secs = int(line.split(":")[-1].strip().split()[0])
            block.append(("start", secs))
        if capture and "Seizure" in line and "End Time" in line:
            secs = int(line.split(":")[-1].strip().split()[0])
            block.append(("end", secs))
    starts = [v for k, v in block if k == "start"]
    ends = [v for k, v in block if k == "end"]
    intervals.extend(zip(starts, ends))
    return intervals


def save_dataset(ds: SpikeDataset, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, spikes=ds.spikes, labels=ds.labels)


def load_saved(path: str) -> SpikeDataset:
    npz = np.load(path)
    return SpikeDataset(spikes=npz["spikes"], labels=npz["labels"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chbmit-dir", type=str, default=None,
                     help="Path to local CHB-MIT directory. Omit for synthetic data.")
    ap.add_argument("--patients", nargs="+", default=["chb01", "chb02"])
    ap.add_argument("--out", type=str, default="artifacts/build/spike_tensors.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.chbmit_dir:
        ds = load_chbmit(args.chbmit_dir, args.patients, seed=args.seed)
    else:
        ds = load_synthetic(seed=args.seed)

    save_dataset(ds, args.out)
    n_sz = int(ds.labels.sum())
    print(f"Saved {len(ds.labels)} windows ({n_sz} seizure, "
          f"{len(ds.labels) - n_sz} non-seizure) -> {args.out}")


if __name__ == "__main__":
    main()
