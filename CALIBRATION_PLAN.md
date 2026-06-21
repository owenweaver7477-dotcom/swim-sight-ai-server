# Known-Distance Calibration Plan

## Status

Known-distance calibration is optional internal tooling. It does not run unless
a `calibration_config` is supplied to the already default-off estimated-metric
path. `ENABLE_ESTIMATED_DRAG` remains `false` by default.

With no calibration input, the worker uses its existing monocular
anthropometric scale estimate and produces the same output as before.

## Supported Input

The worker can derive image scale from two marked image points with a known
real-world separation. Suitable controlled references may include a measured
pool marking or lane reference that lies in the relevant image plane.

```json
{
  "calibration_type": "known_distance",
  "image_points": [
    {"x": 0.25, "y": 0.60},
    {"x": 0.75, "y": 0.60}
  ],
  "real_distance_m": 2.5,
  "coordinate_space": "normalised",
  "image_width": 1920,
  "image_height": 1080
}
```

Pixel coordinates are also supported when positive image dimensions are
provided. Raw image points are used for calculation only and are not echoed in
the callback metric summary.

## Output Language

Valid input is labelled:

> calibrated from marked image distance; drag remains estimated -- not measured

Missing calibration preserves the existing estimated basis. Invalid input
returns a clear internal reason and safely falls back to the existing estimate.
An optional `confidence` value below the internal `0.5` threshold also falls
back instead of being treated as calibrated.

## Important Limits

Known-distance calibration improves image scale only. It does not correct:

- water refraction
- camera perspective or an off-axis reference
- lens distortion
- swimmer occlusion
- pose-landmark error
- movement outside the calibrated image plane

It therefore does not validate biomechanics or hydrodynamics. All resulting
context remains an AI-assisted internal estimate and requires coach review.

## Local Verification

```bash
python3 scripts/test_calibration.py
python3 scripts/test_calibration_core.py
python3 scripts/test_upgrades.py
```

No real footage or calibration image is required for these synthetic checks.
