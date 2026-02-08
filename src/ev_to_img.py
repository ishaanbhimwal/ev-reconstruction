import numpy as np


def events_to_image_binary(x, y, p, height, width):
    """
    Visual event image: gray background, white for p>0, black for p<0.

    Parameters
    ----------
    x, y: event coordinates (int32)
    p:    polarity (-1 / +1) (int8)
    height, width: image size
    """
    img = np.full((height, width), 0.5, dtype=np.float32)
    pos_mask = p > 0
    neg_mask = p < 0

    img[y[pos_mask], x[pos_mask]] = 1.0
    img[y[neg_mask], x[neg_mask]] = 0.0
    return img


def make_event_video_for_range(davis_group, img_idx_start, img_idx_end, dt=0.01):
    """
    Build event-video over frames [img_idx_start, img_idx_end) with window dt.

    Parameters
    ----------
    davis_group: MVSEC DAVIS group (e.g. f["davis"]["left"])
    img_idx_start, img_idx_end: frame index range [start, end)
    dt: window size in seconds for event binning

    Returns
    -------
    event_imgs: list of 2D float32 images in [0,1]
    t_centers:  list of window center timestamps (absolute, seconds)
    """
    event_inds = davis_group["image_raw_event_inds"][:]
    events = davis_group["events"][:]

    start = event_inds[img_idx_start]
    end = event_inds[img_idx_end]

    ev_slice = events[start:end]
    x = ev_slice[:, 0].astype(np.int32)
    y = ev_slice[:, 1].astype(np.int32)
    t_abs = ev_slice[:, 2].astype(np.float64)
    p = ev_slice[:, 3].astype(np.int8)

    image_raw = davis_group["image_raw"]
    _, H, W = image_raw.shape

    t_start = t_abs[0]
    t_end = t_abs[-1]

    event_imgs = []
    t_centers = []

    t_window_start = t_start
    while t_window_start < t_end:
        t_window_end = t_window_start + dt
        mask = (t_abs >= t_window_start) & (t_abs < t_window_end)
        if np.any(mask):
            img = events_to_image_binary(x[mask], y[mask], p[mask], H, W)
            event_imgs.append(img)
            t_centers.append(0.5 * (t_window_start + t_window_end))
        t_window_start = t_window_end

    return event_imgs, t_centers
