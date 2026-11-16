# DentUnity reference model weights

The validation-selected reference model of each dense task (seed 0), as reported in the paper.

| File | Model | Task |
|---|---|---|
| dentunity_iodet_y11n_s0.pt | YOLO11n | Intraoral finding detection (IO-DET) |
| dentunity_pxdet_y11m_s0.pt | YOLO11m | Panoramic detection (PX-DET) |
| dentunity_pxseg_y11s_s0.pt | YOLO11s-seg | Panoramic instance segmentation (PX-SEG) |

Load with `from ultralytics import YOLO; YOLO("dentunity_pxdet_y11m_s0.pt")`. Optimizer state and
local training paths were removed.

**Licence:** fine-tuned from Ultralytics base weights and released under **AGPL-3.0**, which applies
to this `weights/` folder only. The code is MIT. Training images remain under their upstream
publishers' terms; the weights are provided for research use.
