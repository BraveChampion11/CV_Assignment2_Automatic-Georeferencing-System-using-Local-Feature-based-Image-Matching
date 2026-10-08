# Automatic Georeferencing System using Local Feature-based Image Matching
**Course:** Applied Computer Vision (MS-AI) - Assignment 2  **Student:** Tehami Azmat (24i-7616)  **Date:** 2026-10-08

## 1. Objective
Given a georeferenced satellite image and a UAV frame, return the geodetic position (lat, lon) of any clicked pixel of the UAV frame.

## 2. Method
**2.1 Local features.** Keypoints are repeatable image locations (corners/blobs) and descriptors are vectors summarising the neighbourhood so the same point can be recognised in another image.
- *SIFT*: scale-space DoG detector, 128-D gradient-histogram descriptor (float, L2 distance). Invariant to scale and rotation; most robust here.
- *AKAZE*: nonlinear scale space, binary MLDB descriptor (Hamming distance); faster.
- *ORB*: FAST corners + rotated BRIEF, binary; fastest but weakest to large scale change.
- *BRISK*: binary descriptor, scale-aware.
Preprocessing: grayscale + CLAHE, to reduce the illumination/sensor gap between satellite and UAV images. Large images are downscaled for detection and keypoint coordinates are mapped back to full resolution. Reference features are computed once and cached.

**2.2 Matching.** Brute-force kNN (k=2). A match is kept only if best distance < 0.75 x second-best distance (Lowe ratio test), discarding ambiguous matches.

**2.3 Homography + RANSAC.** The ground is approximately planar, so aerial -> reference is a 3x3 homography H with `[x' y' w]^T = H [x y 1]^T`, pixel = (x'/w, y'/w). RANSAC repeatedly fits H to 4 random matches and keeps the model with most inliers (reprojection error < 5 px), which rejects remaining wrong matches. A result with fewer than 12 inliers is flagged unreliable.

**2.4 Pixel -> geographic coordinate.** The reference GeoTIFF stores an affine geotransform (origin + pixel size). For projected CRSs the map coordinate is transformed to WGS84. This is equivalent to interpolating the lat/lon of the pixel grid.

**2.5 Rotation.** SIFT/AKAZE/ORB/BRISK are rotation invariant; a manual or automatic 90-degree rotation option is nevertheless provided, applied before extraction so the clicked pixel and H share the same frame.

## 3. Implementation
Python, OpenCV, rasterio, PyQt5 (`ags.py`). Two synchronized viewers (aerial, reference), method and rotation selectors, footprint overlay, inlier-match viewer, CSV logging.

## 4. Results
*(Fill in after running; add screenshots under `docs/`.)*

| Image | Method | Good matches | Inliers | Time (s) | Result OK? |
|---|---|---|---|---|---|
| aerial1 | SIFT | | | | |
| aerial1 | AKAZE | | | | |
| ... | | | | | |

Screenshots: click result (aerial + reference), inlier matches, one failure case.

## 5. Discussion
*(Fill in with your observations. Points worth checking:)*
- Which method gave the most inliers and the most reliable lat/lon? Which was fastest?
- Which images failed and why (low texture, large scale/viewpoint change, different season/illumination)?
- Accuracy: pick a landmark, compare returned lat/lon with its true location on a map.
- Limits: planar assumption fails with terrain relief or oblique views; error grows far from matched points.

## 6. Conclusion
*(2-3 sentences.)*