# ev-reconstruction

Basic implementation of event-based image reconstruction. Tested with the [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset. For quick testing, download a sequence and place it as `data/indoor_flying1_data.hdf5`.

This project:

- loads DAVIS events + frames from MVSEC
- bins events into fixed time windows (e.g. 10 ms)
- converts each bin into an image (gray bg, white/black events)
- aligns event-images with DAVIS intensity frames in time
- computes simple similarity metrics (SSIM and MSE)
- writes a side-by-side GIF to visualize the reconstruction

<!-- ![Event vs frame](./res/event_vs_frame.gif) -->

<p align="center">
  <img src="./res/event_vs_frame.gif" width="800" alt="Event vs frame">
</p>
