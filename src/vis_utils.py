import numpy as np
import imageio


def make_side_by_side_gif(event_imgs, frame_imgs, out_path, fps=30):
    """
    Save side-by-side GIF: [event | frame] per time step.

    Parameters
    ----------
    event_imgs, frame_imgs: lists of 2D float32 images in [0,1]
    out_path: output GIF path
    fps: frames per second

    Returns
    -------
    None
    """
    frames = []
    for e, r in zip(event_imgs, frame_imgs):
        e8 = (e * 255).astype(np.uint8)
        r8 = (r * 255).astype(np.uint8)
        pair = np.concatenate([e8, r8], axis=1)
        pair_rgb = np.stack([pair] * 3, axis=-1)
        frames.append(pair_rgb)
    imageio.mimsave(out_path, frames, fps=fps)
