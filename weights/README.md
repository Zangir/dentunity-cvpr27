# DentUnity reference model weights

| File | Model | Task |
|---|---|---|
| dentunity_pxdet_y11s_s0.pt | YOLO11s | Panoramic tooth/finding detection (PX-DET), seed 0 |
| dentunity_pxseg_y11s_s0.pt | YOLO11s-seg | Panoramic instance segmentation (PX-SEG), seed 0 |
| dentunity_iodet_y11s_s0.pt | YOLO11s | Intraoral finding detection (IO-DET), seed 0 |

These are snapshots and will be updated with the final selected models.

**Licence:** these weights are fine-tuned from Ultralytics base weights and are released under
**AGPL-3.0** (https://www.gnu.org/licenses/agpl-3.0.html), which applies to this `weights/`
folder only. The code in this repository is MIT. The training images remain under their
upstream publishers' terms (see the dataset card); the weights are provided for research use.
