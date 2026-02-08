import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.ev_to_img import make_event_video_for_range


def compare_event_video_range_to_frames(
    davis_group, img_idx_start, img_idx_end, dt=0.01, max_time_diff=0.01
):
    """
    Compare event-video (binned in dt) to DAVIS frames using SSIM and MSE.

    Parameters
    ----------
    davis_group: MVSEC DAVIS group (e.g. f["davis"]["left"])
    img_idx_start, img_idx_end: frame index range [start, end)
    dt: window size in seconds for event binning
    max_time_diff: max allowed time offset between event-bin and frame

    Returns
    -------
    event_imgs_aligned: list of event-images (2D float32 in [0,1])
    frame_imgs_aligned: list of DAVIS frames (2D float32 in [0,1])
    ssim_vals:          list of SSIM values (float)
    mse_vals:           list of MSE values (float)
    """
    event_imgs, t_centers_abs = make_event_video_for_range(
        davis_group, img_idx_start, img_idx_end, dt
    )

    image_raw = davis_group["image_raw"][:]
    image_raw_ts = davis_group["image_raw_ts"][:]

    event_imgs_aligned = []
    frame_imgs_aligned = []
    ssim_vals = []
    mse_vals = []

    for img_ev, t_center_abs in zip(event_imgs, t_centers_abs):
        frame_idx = int(np.argmin(np.abs(image_raw_ts - t_center_abs)))
        time_diff = abs(image_raw_ts[frame_idx] - t_center_abs)
        if time_diff > max_time_diff:
            continue

        frame = image_raw[frame_idx].astype(np.float32)
        frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)

        m = np.mean((img_ev - frame_norm) ** 2)
        s = ssim(img_ev, frame_norm, data_range=1.0)

        event_imgs_aligned.append(img_ev)
        frame_imgs_aligned.append(frame_norm)
        mse_vals.append(m)
        ssim_vals.append(s)

    return event_imgs_aligned, frame_imgs_aligned, ssim_vals, mse_vals
