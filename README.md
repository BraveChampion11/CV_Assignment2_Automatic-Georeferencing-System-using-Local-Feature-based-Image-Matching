# Automatic Georeferencing System (AGS) - Local Feature Matching

Applied Computer Vision, Assignment 2 (MS-AI, FAST-NUCES Islamabad).
Click any pixel of a UAV image and get its latitude/longitude by matching the image to a georeferenced satellite image.

## Setup (standalone, pip only - no GDAL/conda needed)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements.txt
```
Put `reference.tif` and `aerial1.png ... aerial14.png` next to `ags.py` (or in `data/`), then:
```bash
python ags.py
```

## Usage
1. Choose an aerial image and a method (SIFT / AKAZE / ORB / BRISK).
2. Left-click a pixel on the aerial image (mouse wheel = zoom).
3. Lat/Lon is drawn at the click, on the reference image, and in the bottom bar.
   The green quad on the reference is the aerial footprint. "Show matches" displays the RANSAC inliers.
4. Every click is appended to `click_log.csv`.

## Pipeline
| Step | Implementation |
|---|---|
| 1 Viewer | PyQt5 `QGraphicsView`, reference read with rasterio |
| 2 Click | mouse event -> scene (x, y) |
| 3 Features | detector+descriptor on CLAHE-normalised grayscale; reference features cached in `cache/` |
| 4 Homography | Lowe ratio test (0.75) + `cv2.findHomography(..., RANSAC, 5 px)` |
| 5 Projection | `cv2.perspectiveTransform` |
| 6 Lat/Lon | GeoTIFF affine geotransform (reprojected to WGS84 if CRS is projected) |
| 7 Display | marker + text at click location |

See `REPORT.md` for method, results and discussion.