# ev-reconstruction

Basic implementation of event-based image reconstruction using the [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset. For quick testing, download any sequence and place it as `data/indoor_flying1_data.hdf5`.

Notes:

- Loads DAVIS events + frames from MVSEC.
- Bins events into fixed time windows (e.g. 10 ms).
- Converts each bin into an image (gray bg, white/black events).
- Aligns event-images with DAVIS intensity frames in time.
- Computes simple similarity metrics (SSIM and MSE).
- Writes a side-by-side GIF to visualize the reconstruction.

<p align="center">
  <img src="./res/event_vs_frame.gif" width="800" alt="Event vs frame">
</p>
