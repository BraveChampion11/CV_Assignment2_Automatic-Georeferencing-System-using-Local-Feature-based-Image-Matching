# Automatic Georeferencing System using Local Feature-based Image Matching
**Course:** Applied Computer Vision (MS-AI) - Assignment 2  **Student:** Tehami Azmat (24i-7616)  **Date:** 2026-10-11

## 1. Objective
Given a georeferenced satellite image and a UAV frame, return the geodetic position (lat, lon) of any clicked pixel of the UAV frame.

## 2. Method
**2.1 Local features.** Keypoints are repeatable image locations (corners/blobs) and descriptors are vectors summarising the neighbourhood so the same point can be recognised in another image.
- *SIFT*: scale-space DoG detector, 128-D gradient-histogram descriptor (float, L2 distance). Invariant to scale and rotation.
- *AKAZE*: nonlinear scale space, binary MLDB descriptor (Hamming distance); faster.
- *ORB*: FAST corners + rotated BRIEF, binary; fastest.
- *BRISK*: binary descriptor, scale-aware.

Preprocessing: grayscale + CLAHE, to reduce the illumination/sensor gap between satellite and UAV images. Large images are downscaled for detection and keypoint coordinates are mapped back to full resolution. Reference features are computed once and cached on disk.

**2.2 Matching.** Brute-force kNN (k=2). A match is kept only if best distance < 0.75 x second-best distance (Lowe ratio test), discarding ambiguous matches. The reference produces more than 65,535 descriptors, which exceeds OpenCV's brute-force matcher limit, so reference descriptors are matched in chunks and the two best candidates are merged (identical result to a single pass).

**2.3 Homography + RANSAC.** The ground is approximately planar, so aerial -> reference is a 3x3 homography H with `[x' y' w]^T = H [x y 1]^T`, pixel = (x'/w, y'/w). RANSAC repeatedly fits H to 4 random matches and keeps the model with most inliers (reprojection error < 5 px), which rejects remaining wrong matches. A result with fewer than 12 inliers is flagged unreliable.

**2.4 Pixel -> geographic coordinate.** The reference GeoTIFF stores an affine geotransform (origin + pixel size). For projected CRSs the map coordinate is transformed to WGS84. This is equivalent to interpolating the lat/lon of the pixel grid.

**2.5 Rotation.** SIFT/AKAZE/ORB/BRISK are rotation invariant; a manual or automatic 90-degree rotation option is nevertheless provided, applied before extraction so the clicked pixel and H share the same frame.

## 3. Implementation
Python, OpenCV, rasterio, PyQt5 (`ags.py`). Two synchronized viewers (aerial, reference), method and rotation selectors, footprint overlay, inlier-match viewer, CSV logging. `benchmark.py` runs all images and methods and writes `results.csv` and match images to `docs/`.

Dataset: `reference1Km.tif` (4026 x 4026 px, 3 bands, EPSG:4326, about 1 km x 1 km) and 14 aerial frames (`aerial1.png` ... `aerial14.png`, roughly 640 x 850 px).

## 4. Results
All 14 aerial images were matched against the reference with all four methods (56 runs, `benchmark.py`). Table cells are *good matches after the ratio test / RANSAC inliers*. A match is considered reliable only with at least 12 inliers.

| Image | SIFT | AKAZE | ORB | BRISK |
|---|---|---|---|---|
| aerial1 | 14 / 5 | 12 / 4 | 67 / 5 | 10 / 4 |
| aerial2 | 14 / 4 | 11 / 4 | 57 / 5 | 6 / 4 |
| aerial3 | 6 / 4 | 10 / 4 | 57 / 1 | 7 / 4 |
| aerial4 | 0 / 0 (fail) | 6 / 4 | 69 / 5 | 7 / 4 |
| aerial5 | 12 / 4 | 12 / 4 | 78 / 6 | 5 / 4 |
| aerial6 | 12 / 5 | 10 / 4 | 53 / 4 | 7 / 4 |
| aerial7 | 12 / 5 | 6 / 4 | 58 / 5 | 12 / 4 |
| aerial8 | 4 / 4 | 7 / 4 | 58 / 5 | 6 / 4 |
| aerial9 | 9 / 4 | 9 / 4 | 58 / 5 | 6 / 4 |
| aerial10 | 15 / 4 | 10 / 5 | 63 / 5 | 6 / 4 |
| aerial11 | 32 / 5 | 21 / 7 | 83 / 6 | 14 / 5 |
| aerial12 | 27 / 5 | 12 / 5 | 69 / 3 | 13 / 5 |
| aerial13 | 21 / 7 | 17 / 6 | 65 / 6 | 16 / 6 |
| aerial14 | 19 / 5 | 19 / 6 | 74 / 5 | 17 / 5 |

Summary:
- **No run reached the reliability threshold.** All 56 runs were flagged unreliable (SIFT on aerial4 found no matches at all). The maximum was 7 inliers (SIFT on aerial13, AKAZE on aerial11).
- A homography is defined by 4 point pairs, so 4-7 inliers is the minimum possible fit and is consistent with chance agreement, not a true alignment.
- Run time per image (reference features cached): SIFT 10.8-30.3 s, AKAZE 1.2-3.6 s, ORB 1.6-3.2 s, BRISK 3.6-16.2 s.
- SIFT and the other methods gave somewhat more good matches on aerial11-14 (SIFT: 19-32) than on aerial1-10 (SIFT: 0-15), but still without a consistent geometric fit.
- ORB reports the most good matches (53-83) but its inliers stay at 1-6, which indicates mostly wrong matches that happen to pass a weak ratio test.

Because no match was reliable, the latitude/longitude returned for a click is not trustworthy, and a positional accuracy comparison against a known landmark could not be made.


## 5. Discussion
**Why the matching fails.** Looking at the data, the aerial frames and the reference differ in several ways that local features are sensitive to:
- *Appearance gap.* The aerial frames are hazy, low-contrast video frames in brown tones, while the reference is a sharp, dark, saturated satellite image. Local gradient patterns around the same corner differ between the two.
- *Viewpoint.* The aerial frames are oblique: buildings show their facades and roofs, and the scale changes across the image. The reference is a near-nadir view. A single homography only fits flat ground, and the oblique view with 3D buildings breaks that assumption.
- *Rotation.* The aerial frames are rotated by a large angle relative to the north-up reference (estimated visually at roughly 60-70 degrees for aerial1). The detectors are rotation invariant, but the descriptors become less reliable as other differences add up.
- *Repeated structure.* The scene contains many identical apartment blocks. The ratio test rejects a correct match when a second, very similar block exists, and wrong matches onto neighbouring blocks are accepted instead.
- *Overlays.* The frames contain a crosshair (and text in some frames), which adds clutter near the image centre.

**Methods.** SIFT gave the most good matches among the descriptors that use the ratio test sensibly (up to 32), at the highest run time. AKAZE and ORB are several times faster but did not produce a better alignment. No method gave a reliable result, so no ranking by accuracy is possible from these runs.

**Limits of the approach.** The planar-homography model is only valid for flat terrain and near-nadir views. For oblique frames of tall buildings it can fit at best a part of the image.

**Reliability check.** The system reports failure explicitly (inlier threshold of 12) rather than returning a wrong coordinate, which is the intended behaviour when matching is not trustworthy.

## 6. Conclusion
The AGS pipeline (viewer, click capture, SIFT/AKAZE/ORB/BRISK features, ratio test, RANSAC homography, geotransform to lat/lon, overlay) is implemented and runs on the full dataset. On the provided data, however, standard local-feature matching against the satellite reference did not produce a reliable homography for any of the 14 frames, because of the haze, the oblique viewpoint, the large rotation and the repeated building pattern. The system correctly flags these cases as unreliable.

**Possible improvements (future work).**
- Pre-align the frames (rotate to north-up and correct perspective) before feature extraction, as the assignment allows.
- Restrict matching to a local window of the reference once a coarse location is known, and use a looser ratio test with a rotation-and-scale (similarity) RANSAC model, which needs only 2 correct matches per sample instead of 4. A preliminary offline test on aerial1 suggested a consistent rotation of about 71-74 degrees and a scale of about 1.1, but this was not verified visually.
- Try learned local features (for example SuperPoint with LightGlue) that are more robust to appearance and viewpoint changes.