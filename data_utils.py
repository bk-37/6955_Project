"""
data_utils.py
─────────────
Loads and aligns OpenSim/OpenCap files for a given subject + trial:

  data/subject{N}/
    EMGData/{trial}_EMG.sto              → muscle activations (expert actions)
    ForceData/{trial}_forces.mot         → ground reaction forces (optional)
    OpenSimData/{source}/IK/{trial}.mot  → joint angles (state observations)

source can be: "Mocap", "Video/HRNet/2-cameras", "Video/HRNet/3-cameras",
               "Video/OpenPose_default", "Video/OpenPose_highAccuracy"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader


# ── path helpers ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"


def get_ik_path(subject: str, trial: str, source: str = "Mocap") -> Path:
    """
    Returns path to the IK .mot file for a given subject/trial/source.

    source examples:
      "Mocap"
      "Video/HRNet/2-cameras"
      "Video/HRNet/3-cameras"
      "Video/OpenPose_default"
      "Video/OpenPose_highAccuracy"
    """
    return DATA_DIR / subject / "OpenSimData" / source / "IK" / f"{trial}.mot"


def get_emg_path(subject: str, trial: str) -> Path:
    return DATA_DIR / subject / "EMGData" / f"{trial}_EMG.sto"


def get_grf_path(subject: str, trial: str) -> Path:
    return DATA_DIR / subject / "ForceData" / f"{trial}_forces.mot"


# ── OpenSim parser ───────────────────────────────────────────────────────────

def parse_opensim(path: Path) -> pd.DataFrame:
    """Read any OpenSim .mot/.sto file, skipping the variable-length header."""
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            skip = i + 1
            break
    df = pd.read_csv(path, sep=r"\s+", skiprows=skip, engine="python")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


# ── signal processing ────────────────────────────────────────────────────────

def lowpass(signal: np.ndarray, cutoff_hz: float = 6.0, fs: float = 100.0,
            order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass filter (applied column-wise)."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, signal, axis=0)


def clip_and_normalize_emg(emg: np.ndarray) -> np.ndarray:
    """Clip static-optimisation residuals to [0, 1]."""
    return np.clip(emg, 0.0, 1.0)


def deg2rad(arr: np.ndarray) -> np.ndarray:
    return np.deg2rad(arr)


def compute_velocity(angles: np.ndarray, dt: float) -> np.ndarray:
    """Central differences, edges use forward/backward."""
    return np.gradient(angles, dt, axis=0)


# ── column definitions ───────────────────────────────────────────────────────

# EMG channels → MyoSuite actuator names
MYOSUITE_MUSCLE_MAP = {
    "soleus_l_activation"  : "soleus_l",
    "gasmed_l_activation"  : "gaslat_l",
    "tibant_l_activation"  : "tibant_l",
    "recfem_l_activation"  : "recfem_l",
    "vasmed_l_activation"  : "vasmed_l",
    "vaslat_l_activation"  : "vaslat_l",
    "semiten_l_activation" : "semiten_l",
    "bflh_l_activation"    : "bflh_l",
    "glmed1_l_activation"  : "glmed1_l",
    "soleus_r_activation"  : "soleus_r",
    "gasmed_r_activation"  : "gaslat_r",
    "vasmed_r_activation"  : "vasmed_r",
    "vaslat_r_activation"  : "vaslat_r",
    "semiten_r_activation" : "semiten_r",
    "bflh_r_activation"    : "bflh_r",
    "glmed1_r_activation"  : "glmed1_r",
}

EMG_COLS  = list(MYOSUITE_MUSCLE_MAP.keys())
N_MUSCLES = len(EMG_COLS)

IK_ROTATIONAL_COLS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]
IK_TRANSLATION_COLS = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
IK_ALL_COLS = IK_ROTATIONAL_COLS + IK_TRANSLATION_COLS

GRF_COLS = [
    "R_ground_force_vy", "L_ground_force_vy",
    "R_ground_force_vx", "L_ground_force_vx",
    "R_ground_force_vz", "L_ground_force_vz",
]


# ── main loader ───────────────────────────────────────────────────────────────

class ExpertData:
    """
    Container for aligned, processed expert trajectories for one subject/trial.

    Attributes
    ----------
    states   : (T, S)  float32  – IK joint angles [rad] + angular velocities [rad/s]
    actions  : (T, A)  float32  – muscle activations ∈ [0, 1]
    grf      : (T, 6)  float32  – ground reaction forces (None if use_grf=False)
    time     : (T,)    float64
    dt, T, S, A
    """

    def __init__(
        self,
        subject:       str,
        trial:         str,
        source:        str   = "Mocap",
        use_grf:       bool  = False,
        add_vel:       bool  = True,
        smooth_emg_hz: float = 10.0,
        smooth_ik_hz:  float = 6.0,
    ):
        ik_path  = get_ik_path(subject, trial, source)
        emg_path = get_emg_path(subject, trial)
        grf_path = get_grf_path(subject, trial)

        ik_raw  = parse_opensim(ik_path)
        emg_raw = parse_opensim(emg_path)

        t_common = ik_raw["time"].values
        dt = float(np.median(np.diff(t_common)))
        fs = 1.0 / dt

        # IK angles → radians
        ik_angles = deg2rad(ik_raw[IK_ROTATIONAL_COLS].values.astype(np.float32))
        ik_trans  = ik_raw[IK_TRANSLATION_COLS].values.astype(np.float32)

        if smooth_ik_hz and smooth_ik_hz < fs / 2:
            ik_angles = lowpass(ik_angles, cutoff_hz=smooth_ik_hz, fs=fs).astype(np.float32)

        if add_vel:
            ik_vel  = compute_velocity(ik_angles, dt).astype(np.float32)
            ik_feat = np.concatenate([ik_angles, ik_vel], axis=1)
        else:
            ik_feat = ik_angles

        # EMG activations — resample to IK time grid if needed
        emg_t    = emg_raw["time"].values
        emg_vals = emg_raw[EMG_COLS].values.astype(np.float64)

        if not np.allclose(emg_t, t_common, atol=1e-6):
            interp = interp1d(emg_t, emg_vals, axis=0, kind="linear",
                              fill_value="extrapolate")
            emg_vals = interp(t_common)

        if smooth_emg_hz and smooth_emg_hz < fs / 2:
            emg_vals = lowpass(emg_vals, cutoff_hz=smooth_emg_hz, fs=fs)

        emg_vals = clip_and_normalize_emg(emg_vals).astype(np.float32)

        # GRF (optional auxiliary feature)
        grf_feat = None
        if use_grf:
            grf_raw  = parse_opensim(grf_path)
            grf_t    = grf_raw["time"].values
            grf_vals = grf_raw[GRF_COLS].values.astype(np.float64)
            interp   = interp1d(grf_t, grf_vals, axis=0, kind="linear",
                                fill_value="extrapolate")
            grf_vals = interp(t_common).astype(np.float32)
            bw_proxy = grf_vals[:, 0].max() + grf_vals[:, 1].max()
            grf_vals /= (bw_proxy + 1e-8)
            grf_feat = grf_vals

        self.states  = ik_feat
        self.actions = emg_vals
        self.grf     = grf_feat
        self.time    = t_common
        self.dt      = dt
        self.T, self.S = ik_feat.shape
        self.A         = emg_vals.shape[1]

        print(f"[ExpertData] {subject}/{trial} ({source})")
        print(f"             T={self.T}, S={self.S}, A={self.A}, dt={dt:.4f}s")


ULRICH_DIR = Path(__file__).parent / "Ulrich_data"

# Bare muscle names in results_forces.sto (no _activation suffix)
ULRICH_MUSCLE_COLS = [c.replace("_activation", "") for c in EMG_COLS]


class UlrichExpertData:
    """
    Loads one subject/trial from the Ulrich static-optimisation dataset.

    Directory layout:
      Ulrich_data/{Subject}/IK/{trial}/output/results_ik.sto
      Ulrich_data/{Subject}/StaticOpt/{trial}/results_forces.sto

    Produces states/actions with identical shape and semantics to ExpertData
    so it can be mixed freely with OpenCap data.
    """

    def __init__(
        self,
        subject:       str,
        trial:         str,
        add_vel:       bool  = True,
        smooth_ik_hz:  float = 6.0,
        smooth_act_hz: float = 10.0,
    ):
        ik_path  = ULRICH_DIR / subject / "IK" / trial / "output" / "results_ik.sto"
        act_path = ULRICH_DIR / subject / "StaticOpt" / trial / "results_forces.sto"

        ik_raw  = parse_opensim(ik_path)
        act_raw = parse_opensim(act_path)

        t_common = ik_raw["time"].values
        dt = float(np.median(np.diff(t_common)))
        fs = 1.0 / dt

        # IK angles → radians (inDegrees=yes in Ulrich IK files)
        ik_angles = deg2rad(ik_raw[IK_ROTATIONAL_COLS].values.astype(np.float32))

        if smooth_ik_hz and smooth_ik_hz < fs / 2:
            ik_angles = lowpass(ik_angles, cutoff_hz=smooth_ik_hz, fs=fs).astype(np.float32)

        if add_vel:
            ik_vel  = compute_velocity(ik_angles, dt).astype(np.float32)
            ik_feat = np.concatenate([ik_angles, ik_vel], axis=1)
        else:
            ik_feat = ik_angles

        # Static-opt activations — resample to IK time grid if needed
        act_t    = act_raw["time"].values
        act_vals = act_raw[ULRICH_MUSCLE_COLS].values.astype(np.float64)

        if not np.allclose(act_t, t_common, atol=1e-4):
            interp   = interp1d(act_t, act_vals, axis=0, kind="linear",
                                fill_value="extrapolate")
            act_vals = interp(t_common)

        if smooth_act_hz and smooth_act_hz < fs / 2:
            act_vals = lowpass(act_vals, cutoff_hz=smooth_act_hz, fs=fs)

        act_vals = clip_and_normalize_emg(act_vals).astype(np.float32)

        self.states  = ik_feat
        self.actions = act_vals
        self.grf     = None
        self.time    = t_common
        self.dt      = dt
        self.T, self.S = ik_feat.shape
        self.A         = act_vals.shape[1]

        print(f"[UlrichExpertData] {subject}/{trial}")
        print(f"                   T={self.T}, S={self.S}, A={self.A}, dt={dt:.4f}s")


def load_ulrich_multi(
    subjects: list[str],
    trials:   list[str],
    **kwargs,
) -> "UlrichExpertData":
    """
    Load and concatenate Ulrich data across subjects/trials.
    Skips missing files gracefully, same as load_multi.
    """
    parts = []
    for subj in subjects:
        for trial in trials:
            try:
                parts.append(UlrichExpertData(subj, trial, **kwargs))
            except FileNotFoundError as e:
                print(f"[load_ulrich_multi] skipping {subj}/{trial}: {e}")

    if not parts:
        raise RuntimeError("No Ulrich data loaded — check subject/trial names.")

    combined = parts[0]
    combined.states  = np.concatenate([p.states  for p in parts], axis=0)
    combined.actions = np.concatenate([p.actions for p in parts], axis=0)
    combined.T       = combined.states.shape[0]
    return combined


def load_multi(
    subjects: list[str],
    trials:   list[str],
    source:   str  = "Mocap",
    **kwargs,
) -> "ExpertData":
    """
    Concatenate ExpertData across multiple subjects and/or trials.
    Returns a single ExpertData-like object with combined arrays.
    """
    parts = []
    for subj in subjects:
        for trial in trials:
            try:
                parts.append(ExpertData(subj, trial, source, **kwargs))
            except FileNotFoundError as e:
                print(f"[load_multi] skipping {subj}/{trial}: {e}")

    if not parts:
        raise RuntimeError("No data loaded — check subject/trial names and paths.")

    combined = parts[0]
    combined.states  = np.concatenate([p.states  for p in parts], axis=0)
    combined.actions = np.concatenate([p.actions for p in parts], axis=0)
    if all(p.grf is not None for p in parts):
        combined.grf = np.concatenate([p.grf for p in parts], axis=0)
    combined.T = combined.states.shape[0]
    return combined


# ── PyTorch dataset ───────────────────────────────────────────────────────────

class GAILDataset(Dataset):
    """
    Wraps ExpertData as a PyTorch Dataset.
    Each item: (state, action, next_state).
    """

    def __init__(self, expert: ExpertData):
        self.states      = torch.from_numpy(expert.states[:-1])
        self.actions     = torch.from_numpy(expert.actions[:-1])
        self.next_states = torch.from_numpy(expert.states[1:])

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]


def make_dataloaders(
    expert:     ExpertData,
    batch_size: int   = 32,
    val_frac:   float = 0.1,
    seed:       int   = 42,
):
    """Split into train/val and return DataLoaders."""
    dataset = GAILDataset(expert)
    n       = len(dataset)
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=gen
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
