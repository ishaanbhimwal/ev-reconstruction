# create_subset.py

import h5py
import numpy as np
from pathlib import Path

SRC_PATH = Path("data/indoor_flying1_first_500.hdf5")
DST_PATH = Path("data/indoor_flying1_first_300.hdf5")
N_FRAMES = 300  # number of DAVIS frames to keep


def subset_davis_group(src_group, dst_group, n_frames):
    """
    Copy first n_frames of DAVIS data (left or right) into dst_group.
    Keeps:
      - events belonging to those frames
      - image_raw, image_raw_ts, image_raw_event_inds
      - imu / imu_ts fully (for simplicity)

    """
    # Copy full imu data (optional; you can drop if not needed)
    for name in ["imu", "imu_ts"]:
        if name in src_group:
            src_group.copy(name, dst_group, name=name)

    # Load full indexing + events
    image_raw = src_group["image_raw"][:]            # (N, H, W)
    image_raw_ts = src_group["image_raw_ts"][:]      # (N,)
    event_inds = src_group["image_raw_event_inds"][:]  # (N+1,)
    events = src_group["events"][:]                  # (N_events, 4)

    n_frames_available = image_raw.shape[0]
    n_keep = min(n_frames, n_frames_available)

    # Frame subset
    image_raw_subset = image_raw[:n_keep]
    image_raw_ts_subset = image_raw_ts[:n_keep]

    # Event subset: all events up to frame n_keep-1 => use event_inds[n_keep]
    start_event_idx = event_inds[0]
    end_event_idx = event_inds[n_keep]  # exclusive
    events_subset = events[start_event_idx:end_event_idx]

    # Recompute image_raw_event_inds relative to new events array
    # We shift indices so that events_subset[0] corresponds to index 0
    event_inds_subset = event_inds[: (n_keep + 1)] - start_event_idx

    # Write to dst_group
    dst_group.create_dataset("image_raw", data=image_raw_subset, compression="gzip")
    dst_group.create_dataset("image_raw_ts", data=image_raw_ts_subset, compression="gzip")
    dst_group.create_dataset("image_raw_event_inds", data=event_inds_subset, compression="gzip")
    dst_group.create_dataset("events", data=events_subset, compression="gzip")


def main():
    DST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(SRC_PATH, "r") as f_src, h5py.File(DST_PATH, "w") as f_dst:
        # Create /davis group
        davis_dst = f_dst.create_group("davis")

        # Subset left and right (if present)
        for cam in ["left", "right"]:
            if cam in f_src["davis"]:
                print(f"Subsetting davis/{cam} ...")
                src_group = f_src["davis"][cam]
                dst_group = davis_dst.create_group(cam)
                subset_davis_group(src_group, dst_group, N_FRAMES)

        # Optionally copy velodyne as-is, or skip
        if "velodyne" in f_src:
            print("Copying /velodyne (full, no subset)...")
            f_src.copy("velodyne", f_dst, name="velodyne")

    print("Wrote subset to", DST_PATH)


if __name__ == "__main__":
    main()
