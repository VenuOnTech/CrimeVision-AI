# Labelled research dataset format

Do not use `make_video.py` synthetic output as training, calibration, or test evidence.
Create separate CSV files for `train`, `validation`, `calibration`, and `test`.
Each CSV must contain these columns:

```csv
image_path,statement,label,camera_id,environment_id,source_recording_id
samples/frame_0001.jpg,"A red vehicle is parked beside the building",MATCH,CAMERA_A,ENV_CAPTURE_A_DAYLIGHT,RECORDING_0001
samples/frame_0032.jpg,"A blue vehicle is parked beside the building",CONTRADICTION,CAMERA_B,ENV_CAPTURE_B_NIGHT,RECORDING_0002
```

- `image_path`: path to a genuine evidence image, absolute or relative to the CSV.
- `statement`: the human-provided statement paired with that image.
- `label`: independently assigned `MATCH` or `CONTRADICTION` label. Define `MATCH` as the visible frame supporting the specific claim and `CONTRADICTION` as affirmative visible evidence against it. Exclude unverifiable or ambiguous pairs from this binary training set; do not guess.
- `camera_id`: stable source camera identifier. The scripts reject camera overlap between train/validation/calibration/test.
- `environment_id`: stable identifier for the specific capture-environment group. Do not use a broad repeated label such as `DAY` or `NIGHT` if it recurs across unrelated captures, because the scripts require environment groups to be disjoint between splits. Keep descriptive condition categories in a separate manifest if needed.
- `source_recording_id`: stable identifier for the original source recording. Every frame extracted from one video must share this ID. The scripts reject source-recording overlap across train, validation, calibration and test splits so adjacent frames cannot leak across partitions.

For videos, export frames with source-video ID and timestamps retained in a separate manifest. Do not treat multiple adjacent frames from one source video as independent camera samples. Ideally group all frames from the same source video under a common `camera_id` or add a stricter `source_video_id` grouping rule before training.

The test CSV must remain untouched until model training and threshold calibration are complete. Include both classes in every split. Calibration needs at least the configured number of accepted examples per class; the default is five, but more diverse labelled samples are strongly preferable.
