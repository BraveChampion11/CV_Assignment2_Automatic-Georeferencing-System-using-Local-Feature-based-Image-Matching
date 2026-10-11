"""
Runs every aerial image against the reference with every available method and saves:
  results.csv                       - table for the report
  docs/footprints_<METHOD>.jpg      - all aerial footprints drawn on the reference
  docs/matches_<image>_SIFT.jpg     - RANSAC inlier matches (SIFT) for each image
Usage:  python benchmark.py
"""
import os, re, csv, glob, time
import numpy as np
import cv2
import ags

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)

ref_path = None
for d in (HERE, os.path.join(HERE, "dataset"), os.path.join(HERE, "data")):
    r = sorted(glob.glob(os.path.join(d, "reference*.tif*")))
    if r:
        ref_path = r[0]
        break
img_dir = next(d for d in (os.path.join(HERE, "dataset"), os.path.join(HERE, "data"), HERE)
               if glob.glob(os.path.join(d, "aerial*")))
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(s))]
images = sorted([f for f in glob.glob(os.path.join(img_dir, "aerial*"))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))], key=nat)

ref = ags.GeoReference(ref_path)
methods = ags.available_methods()
print("Reference:", ref_path, ref.w, "x", ref.h)
print("Images:", len(images), "| Methods:", methods)

rows, foot = [], {m: [] for m in methods}
for m in methods:
    print("caching reference features for", m, "...")
    ags.reference_features(ref.path, ref.gray, m)          # cached; excluded from timing

for path in images:
    name = os.path.splitext(os.path.basename(path))[0]
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    for m in methods:
        pr, dr = ags.reference_features(ref.path, ref.gray, m)
        t0 = time.time()
        pa, da = ags.extract_features(gray, m, ags.AERIAL_MAX_DIM)
        res = ags.estimate_homography(pa, da, pr, dr, m)
        dt = time.time() - t0
        if res is None:
            rows.append([name, m, len(pa), 0, 0, round(dt, 2), "", "", "FAIL"])
            continue
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, res["H"]).reshape(-1, 2)
        convex = cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2))
        c = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), res["H"])[0, 0]
        lat, lon = ref.pixel_to_latlon(float(c[0]), float(c[1]))
        ok = res["n_inl"] >= ags.MIN_INLIERS and convex
        rows.append([name, m, len(pa), res["n_good"], res["n_inl"], round(dt, 2),
                     "{:.6f}".format(lat), "{:.6f}".format(lon), "OK" if ok else "UNRELIABLE"])
        if ok:
            foot[m].append((name, quad))
        if m == "SIFT":
            vis = ags.draw_matches(rgb, ref.rgb, res["src"], res["dst"])
            cv2.imwrite(os.path.join(DOCS, "matches_{}_SIFT.jpg".format(name)),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(name, [r[4] for r in rows if r[0] == name])

with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", "method", "aerial_keypoints", "good_matches", "inliers", "time_s",
                 "center_lat", "center_lon", "status"])
    wr.writerows(rows)

s = min(1.0, 1800.0 / max(ref.w, ref.h))
for m, quads in foot.items():
    img = cv2.resize(ref.rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).copy()
    for name, q in quads:
        p = (q * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], True, (0, 255, 0), 2)
        cv2.putText(img, name.replace("aerial", ""), tuple(p[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(DOCS, "footprints_{}.jpg".format(m)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print("\n| Image | Method | Good matches | Inliers | Time (s) | Status |\n|---|---|---|---|---|---|")
for r in rows:
    print("| {} | {} | {} | {} | {} | {} |".format(r[0], r[1], r[3], r[4], r[5], r[8]))
print("\nSaved results.csv and docs/*.jpg")