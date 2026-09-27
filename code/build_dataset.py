"""
build_dataset.py  --  CogBeacon -> per-round multimodal feature table.

Usage:  python code/build_dataset.py /path/to/cogbeacon [data/dataset.npz]

LABEL / FEATURE SEPARATION
--------------------------
Labels are read ONLY from fatigue_self_report/*.csv (load_press_counts).
Features are read ONLY from eeg/, face_keypoints/, user_performance/.
The two sides are separate functions that share no state. The closing
assertion is a name check: it fails the build if any feature name contains a
label-related substring. It guards against a self-report column being added
as a feature; it cannot detect behavioural traces of the reporting act in the
feature streams, which is what MTEP step 6b (event-unit exclusion) addresses.

Feature groups (by name prefix): 72 eeg_*, 13 face_*, 12 perf_*.
perf_round_frames (number of video frames in the round, i.e. round duration at
2 fps) is placed in the task/timing group, not the facial group.

Round alignment key: roundID (1-based), consistent across all four folders.
"""

import os, re, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    sys.exit("usage: python code/build_dataset.py /path/to/cogbeacon [out.npz]")
ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "data", "dataset.npz")

BANDS_ABS = ["Aa", "Ab", "Ad", "Ag", "At"]      # absolute band power, 4 ch
BANDS_REL = ["a", "b", "d", "g", "t"]           # relative band power, 4 ch
QUALITY = "h"                                   # signal quality 1(best)-4(worst)

# dlib-68 landmark index groups
R_EYE = list(range(36, 42))
L_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))


# ----------------------------------------------------------------------
# session discovery
# ----------------------------------------------------------------------
def list_sessions(root):
    """Return [(session_name, user_id, day, stimuli, mode), ...]."""
    sessions = []
    for name in sorted(os.listdir(os.path.join(root, "eeg"))):
        m = re.match(r"user_(\d+)(b?)_([vta])_([mo])$", name)
        if not m:
            print(f"  [skip] unparsed session name: {name}")
            continue
        uid, b, stim, mode = m.groups()
        sessions.append(dict(session=name, user=int(uid),
                             day=2 if b else 1, stimuli=stim, mode=mode))
    return sessions


# ----------------------------------------------------------------------
# LABEL SIDE  -- reads fatigue_self_report ONLY
# ----------------------------------------------------------------------
def load_press_counts(root, session):
    """Cumulative button-press count per round. Row i -> roundID i+1."""
    path = os.path.join(root, "fatigue_self_report", session + ".csv")
    vals = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                vals.append(int(float(line)))
    return np.asarray(vals, dtype=int)


def new_press_events(counts):
    """Binary per round: 1 if a NEW press was registered at this round."""
    ev = np.zeros(len(counts), dtype=int)
    ev[1:] = (np.diff(counts) > 0).astype(int)
    ev[0] = 1 if counts[0] > 0 else 0
    return ev


# ----------------------------------------------------------------------
# FEATURE SIDE  -- reads eeg / face_keypoints / user_performance ONLY
# ----------------------------------------------------------------------
def parse_eeg_round(path):
    """Return dict of per-tag arrays for one round file."""
    tags = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            tag, vals = parts[0], parts[1:]
            try:
                v = [float(x) for x in vals]
            except ValueError:
                continue
            tags.setdefault(tag, []).append(v)
    return {k: np.asarray(v, dtype=float) for k, v in tags.items()}


def eeg_features(tags):
    """72 EEG features: 10 band means x 4 ch, 5 relative-band SDs x 4 ch,
    theta/alpha and beta/(theta+alpha) x 4 ch, signal quality x 4 ch."""
    feats, names = [], []

    def push(vec, prefix):
        for i, x in enumerate(vec):
            feats.append(float(x)); names.append(f"eeg_{prefix}_ch{i}")

    for tag in BANDS_ABS + BANDS_REL:
        arr = tags.get(tag)
        if arr is None or arr.size == 0:
            push([np.nan] * 4, f"{tag}_mean")
        else:
            push(np.nanmean(arr, axis=0)[:4], f"{tag}_mean")

    for tag in BANDS_REL:
        arr = tags.get(tag)
        if arr is None or arr.size == 0:
            push([np.nan] * 4, f"{tag}_sd")
        else:
            push(np.nanstd(arr, axis=0)[:4], f"{tag}_sd")

    # engagement / fatigue ratios from relative bands
    def band_mean(tag):
        arr = tags.get(tag)
        if arr is None or arr.size == 0:
            return np.full(4, np.nan)
        return np.nanmean(arr, axis=0)[:4]

    th, al, be = band_mean("t"), band_mean("a"), band_mean("b")
    with np.errstate(divide="ignore", invalid="ignore"):
        push(th / al, "theta_alpha")                 # classic fatigue index
        push(be / (th + al), "engagement")           # engagement index

    q = tags.get(QUALITY)
    push(np.nanmean(q, axis=0)[:4] if q is not None and q.size else [np.nan] * 4,
         "quality")

    return np.asarray(feats), names


def _aspect_ratio(pts):
    """Extent ratio (height / width of the landmark group's bounding extent).

    This is NOT the landmark-distance eye aspect ratio of Soukupova & Cech;
    it is a simpler extent ratio and is named accordingly."""
    if pts.size == 0:
        return np.nan
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return h / w if w > 0 else np.nan


def face_features(session_dir, round_id):
    """Aggregate all frames of one round into 14 per-round values.

    13 are facial (face_*). Extent ratios are scale-free; centroid motion is
    normalised by bounding-box width; bounding-box geometry is in raw pixels
    (suffix _px). The 14th value, the frame count, measures round duration and
    is named perf_round_frames so that it falls in the task/timing group."""
    names = ["face_eye_extent_mean", "face_eye_extent_sd",
             "face_mouth_extent_mean", "face_mouth_extent_sd",
             "face_bbox_w_px", "face_bbox_h_px", "face_bbox_x_px", "face_bbox_y_px",
             "face_motion_mean", "face_motion_sd", "face_motion_max",
             "perf_round_frames", "face_head_x_sd", "face_head_y_sd"]
    if not os.path.isdir(session_dir):
        return np.full(len(names), np.nan), names

    frames = []
    for fn in os.listdir(session_dir):
        m = re.match(r"(\d+)_(\d+)_(\d+)\.npz$", fn)
        if m and int(m.group(2)) == round_id:
            frames.append((int(m.group(3)), os.path.join(session_dir, fn)))
    if not frames:
        return np.full(len(names), np.nan), names
    frames.sort()

    ears, mars, bboxes, centroids = [], [], [], []
    for _, path in frames:
        try:
            a = np.load(path)["arr_0"]
        except Exception:
            continue
        if a.size < 140:
            continue
        kp = a[:136].reshape(68, 2).astype(float)
        bbox = a[136:140].astype(float)
        # scale-normalise by bounding-box width so camera distance cancels
        scale = bbox[2] if bbox[2] > 0 else 1.0
        ears.append(np.nanmean([_aspect_ratio(kp[R_EYE]), _aspect_ratio(kp[L_EYE])]))
        mars.append(_aspect_ratio(kp[MOUTH]))
        bboxes.append(bbox)
        centroids.append(kp.mean(axis=0) / scale)

    if not ears:
        return np.full(len(names), np.nan), names

    ears = np.asarray(ears, float)
    mars = np.asarray(mars, float)
    bboxes = np.asarray(bboxes, float)
    cent = np.asarray(centroids, float)
    motion = (np.linalg.norm(np.diff(cent, axis=0), axis=1)
              if len(cent) > 1 else np.array([0.0]))

    vals = [np.nanmean(ears), np.nanstd(ears), np.nanmean(mars), np.nanstd(mars),
            np.nanmean(bboxes[:, 2]), np.nanmean(bboxes[:, 3]),
            np.nanmean(bboxes[:, 0]), np.nanmean(bboxes[:, 1]),
            np.nanmean(motion), np.nanstd(motion), np.nanmax(motion),
            float(len(ears)), np.nanstd(cent[:, 0]), np.nanstd(cent[:, 1])]
    return np.asarray(vals, float), names


def load_performance(root, session):
    """Per-round task metrics. Returns DataFrame indexed by roundID."""
    matches = [f for f in os.listdir(os.path.join(root, "user_performance"))
               if f.startswith(session + "_")]
    if not matches:
        return None
    df = pd.read_csv(os.path.join(root, "user_performance", matches[0]), sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def performance_features(df, idx, round_id):
    """11 task-performance features for one round (no self-report involved)."""
    names = ["perf_level", "perf_score", "perf_response_time", "perf_correct",
             "perf_question", "perf_persistence", "perf_round_norm",
             "perf_d_nonper_err", "perf_d_per_err", "perf_rt_dev", "perf_acc_run"]
    if df is None or idx >= len(df):
        return np.full(len(names), np.nan), names

    row = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else row

    def num(r, col, default=np.nan):
        try:
            v = r[col]
            if isinstance(v, str):
                v = v.strip()
                if v in ("True", "False"):
                    return 1.0 if v == "True" else 0.0
            return float(v)
        except Exception:
            return default

    rt = num(row, "Time")
    window = df["Time"].iloc[max(0, idx - 9):idx + 1].astype(float)
    rt_dev = (rt - window.mean()) / (window.std() + 1e-6) if len(window) > 1 else 0.0
    acc_win = df["Response"].iloc[max(0, idx - 9):idx + 1]
    acc_run = np.mean([1.0 if str(x).strip() == "True" else 0.0 for x in acc_win])

    vals = [num(row, "Level"), num(row, "Score"), rt, num(row, "Response"),
            num(row, "Question"), num(row, "Persistence"),
            round_id / max(len(df), 1),
            num(row, "NON-PER Errors") - num(prev, "NON-PER Errors"),
            num(row, "PER Errorsn") - num(prev, "PER Errorsn"),
            rt_dev, acc_run]
    return np.asarray(vals, float), names


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------
def build(root):
    sessions = list_sessions(root)
    print(f"Found {len(sessions)} sessions")

    X_rows, meta_rows, y_event = [], [], []
    feature_names = None

    for si, s in enumerate(sessions):
        sess = s["session"]
        eeg_dir = os.path.join(root, "eeg", sess)
        face_dir = os.path.join(root, "face_keypoints", sess)

        try:
            counts = load_press_counts(root, sess)      # LABEL SIDE
        except FileNotFoundError:
            print(f"  [skip] no self-report for {sess}")
            continue
        events = new_press_events(counts)

        perf = load_performance(root, sess)             # FEATURE SIDE

        round_ids = sorted(
            int(fn.split("_")[1]) for fn in os.listdir(eeg_dir)
            if re.match(r"\d+_\d+$", fn))
        eeg_by_round = {}
        for fn in os.listdir(eeg_dir):
            m = re.match(r"(\d+)_(\d+)$", fn)
            if m:
                eeg_by_round[int(m.group(2))] = os.path.join(eeg_dir, fn)

        n_rounds = min(len(counts), len(round_ids))
        for r in range(1, n_rounds + 1):
            if r not in eeg_by_round:
                continue
            ef, en = eeg_features(parse_eeg_round(eeg_by_round[r]))
            ff, fn_ = face_features(face_dir, r)
            pf, pn = performance_features(perf, r - 1, r)

            row = np.concatenate([ef, ff, pf])
            if feature_names is None:
                feature_names = en + fn_ + pn
            X_rows.append(row)
            y_event.append(events[r - 1])
            meta_rows.append((si, s["user"], s["day"], r, n_rounds))

        if (si + 1) % 10 == 0:
            print(f"  processed {si+1}/{len(sessions)} sessions, "
                  f"{len(X_rows)} rounds")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_event, dtype=np.int8)
    meta = np.asarray(meta_rows, dtype=np.int32)

    # ---------------- label/feature name check ----------------
    banned = ("selfreport", "self_report", "press", "fatigue_label",
              "label", "kss", "count")
    for nm in feature_names:
        assert not any(b in nm.lower() for b in banned), \
            f"LEAKAGE: feature '{nm}' references the labelling signal"
    assert len(feature_names) == X.shape[1]
    print("Name check passed: no feature name references the self-report stream.")

    groups = dict(
        eeg=[i for i, n in enumerate(feature_names) if n.startswith("eeg_")],
        face=[i for i, n in enumerate(feature_names) if n.startswith("face_")],
        perf=[i for i, n in enumerate(feature_names) if n.startswith("perf_")],
    )

    np.savez_compressed(
        OUT, X=X, y_event=y, meta=meta,
        feature_names=np.array(feature_names),
        group_eeg=np.array(groups["eeg"]), group_face=np.array(groups["face"]),
        group_perf=np.array(groups["perf"]))

    print(f"\nX={X.shape}  y_event positives={int(y.sum())} "
          f"({100*y.mean():.2f}%)  users={len(set(meta[:,1]))}  "
          f"sessions={len(set(meta[:,0]))}")
    print(f"EEG feats={len(groups['eeg'])}  face={len(groups['face'])}  "
          f"perf={len(groups['perf'])}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    build(ROOT)
