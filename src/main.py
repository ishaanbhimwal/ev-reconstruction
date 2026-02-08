import numpy as np
import h5py
from pathlib import Path

from src.davis_allign import compare_event_video_range_to_frames
from src.vis_utils import make_side_by_side_gif

h5_path = "./data/indoor_flying1_first_300.hdf5"
with h5py.File(h5_path, "r") as f:
    davis_left = f["davis"]["left"]

    img_idx_start = 100  # Not many events b/w 0-100 hence start from 100
    img_idx_end = 300  # 200 frames
    dt = 0.01  # 10 ms per event frame

    event_imgs, frame_imgs, ssim_vals, mse_vals = compare_event_video_range_to_frames(
        davis_left, img_idx_start, img_idx_end, dt=dt, max_time_diff=0.01
    )

    print("Num aligned pairs:", len(event_imgs))
    if ssim_vals:
        print("Mean SSIM:", np.mean(ssim_vals))
        print("Mean MSE:", np.mean(mse_vals))

    out_path = Path("./res/event_vs_frame.gif")
    make_side_by_side_gif(event_imgs, frame_imgs, out_path, fps=30)
    print("Saved GIF to", out_path)
