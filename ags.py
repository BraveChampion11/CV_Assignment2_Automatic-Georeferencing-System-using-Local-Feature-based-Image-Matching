#!/usr/bin/env python3
"""
Automatic Georeferencing System (AGS) using local feature-based image matching
Applied Computer Vision - Assignment 2 (MS-AI, FAST-NUCES Islamabad)

Steps implemented
  1. Viewer: load reference.tif (rasterio) and a folder of aerial images, view both side by side
  2. Click any pixel on the aerial image -> (x, y) is stored
  3. Keypoints + descriptors for aerial and reference (reference features are cached on disk)
  4. Homography H (aerial -> reference) with RANSAC
  5. Project the clicked pixel with H
  6. Convert projected reference pixel to latitude / longitude (GeoTIFF geotransform)
  7. Draw the coordinates at the click location (and on the reference image)

Extras: SIFT / SURF (if available) / AKAZE / ORB / BRISK, manual or automatic 90-degree
rotation, aerial footprint drawn on the reference, inlier-match viewer, CSV click log.
"""
import os
import re
import sys
import csv
import time
import glob
import pickle
import hashlib

import numpy as np
import cv2
import rasterio
from rasterio.warp import transform as warp_transform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
LOG_CSV = os.path.join(HERE, "click_log.csv")

REF_MAX_DIM = 4000      # reference is downscaled to this for feature extraction (coords mapped back)
AERIAL_MAX_DIM = 2000   # same for aerial images
RATIO = 0.75            # Lowe ratio test
RANSAC_THRESH = 5.0     # reprojection threshold (reference pixels)
MIN_INLIERS = 12        # below this the match is considered unreliable


# --------------------------------------------------------------------------------------
# Feature extraction / matching (steps 3 and 4)
# --------------------------------------------------------------------------------------
def available_methods():
    methods = ["SIFT", "AKAZE", "ORB", "BRISK"]
    try:
        cv2.xfeatures2d.SURF_create()
        methods.insert(1, "SURF")      # only in opencv-contrib builds with nonfree enabled
    except Exception:
        pass
    return methods


def create_detector(method):
    """Returns (detector, matcher norm)."""
    if method == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2
    if method == "SURF":
        return cv2.xfeatures2d.SURF_create(400), cv2.NORM_L2
    if method == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if method == "ORB":
        return cv2.ORB_create(nfeatures=20000, fastThreshold=5), cv2.NORM_HAMMING
    if method == "BRISK":
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    raise ValueError(method)


def extract_features(gray, method, max_dim):
    """Keypoints (as Nx2 array in ORIGINAL image coordinates) and descriptors."""
    s = min(1.0, max_dim / float(max(gray.shape[:2])))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)  # reduces illumination gap
    det, _ = create_detector(method)
    kps, des = det.detectAndCompute(g, None)
    if des is None or len(kps) < 4:
        return np.zeros((0, 2), np.float32), None
    pts = np.array([k.pt for k in kps], np.float32) / s
    return pts, des


def reference_features(ref_path, gray, method):
    """Pre-computes and caches reference keypoints/descriptors on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    st = os.stat(ref_path)
    key = hashlib.md5(
        "{}|{}|{}|{}|{}".format(ref_path, st.st_size, st.st_mtime, REF_MAX_DIM, method).encode()
    ).hexdigest()[:12]
    f = os.path.join(CACHE_DIR, "ref_{}_{}.pkl".format(method, key))
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    data = extract_features(gray, method, REF_MAX_DIM)
    with open(f, "wb") as fh:
        pickle.dump(data, fh)
    return data


def estimate_homography(pa, da, pr, dr, method):
    """Ratio-test matching + RANSAC homography mapping aerial -> reference."""
    if da is None or dr is None:
        return None
    _, norm = create_detector(method)
    knn = cv2.BFMatcher(norm).knnMatch(da, dr, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 4:
        return None
    src = np.float32([pa[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([pr[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH, maxIters=10000, confidence=0.995)
    if H is None:
        return None
    mask = mask.ravel().astype(bool)
    return dict(H=H, n_good=len(good), n_inl=int(mask.sum()),
                src=src[mask].reshape(-1, 2), dst=dst[mask].reshape(-1, 2))


def draw_matches(img_a, img_r, src, dst, max_draw=150):
    sa = min(1.0, 900.0 / max(img_a.shape[:2]))
    sr = min(1.0, 900.0 / max(img_r.shape[:2]))
    a = cv2.resize(img_a, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    r = cv2.resize(img_r, None, fx=sr, fy=sr, interpolation=cv2.INTER_AREA)
    idx = np.linspace(0, len(src) - 1, min(len(src), max_draw)).astype(int)
    kpa = [cv2.KeyPoint(float(src[i][0] * sa), float(src[i][1] * sa), 5) for i in idx]
    kpr = [cv2.KeyPoint(float(dst[i][0] * sr), float(dst[i][1] * sr), 5) for i in idx]
    ms = [cv2.DMatch(i, i, 0) for i in range(len(idx))]
    return cv2.drawMatches(a, kpa, r, kpr, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# --------------------------------------------------------------------------------------
# Reference image + geo information (rasterio, no GDAL install needed) (step 6)
# --------------------------------------------------------------------------------------
class GeoReference:
    def __init__(self, path):
        self.path = path
        with rasterio.open(path) as ds:
            self.w, self.h = ds.width, ds.height
            self.transform = ds.transform          # affine: pixel -> map coordinates
            self.crs = ds.crs
            n = 3 if ds.count >= 3 else 1
            img = ds.read(list(range(1, n + 1)))   # (bands, h, w)
        img = np.transpose(img, (1, 2, 0)) if n == 3 else img[0]
        if img.dtype != np.uint8:      # stretch 16-bit / float data to 8-bit
            f = img.astype(np.float32)
            lo, hi = np.percentile(f, 2), np.percentile(f, 98)
            img = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.rgb = np.ascontiguousarray(img)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self.needs_reproject = self.crs is not None and self.crs.to_epsg() != 4326

    def pixel_to_latlon(self, px, py):
        """Affine geotransform (equivalent to bilinear interpolation of the corner coords)."""
        X, Y = self.transform * (px + 0.5, py + 0.5)     # centre of pixel
        if self.needs_reproject:
            xs, ys = warp_transform(self.crs, "EPSG:4326", [X], [Y])
            X, Y = xs[0], ys[0]
        return Y, X   # lat, lon


# --------------------------------------------------------------------------------------
# GUI widgets (steps 1, 2, 7)
# --------------------------------------------------------------------------------------
def to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))


class ImageView(QtWidgets.QGraphicsView):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, title):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setBackgroundBrush(QtGui.QColor(30, 30, 30))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setToolTip(title + "  |  wheel = zoom, left click = pick pixel")
        self.pix = None
        self.items_extra = []

    def set_image(self, rgb):
        self.scene().clear()
        self.items_extra = []
        self.pix = self.scene().addPixmap(to_pixmap(rgb))
        self.setSceneRect(self.pix.boundingRect())
        self.resetTransform()
        self.fitInView(self.pix, Qt.KeepAspectRatio)

    def clear_markers(self):
        for it in self.items_extra:
            self.scene().removeItem(it)
        self.items_extra = []

    def add_marker(self, x, y, text, color=Qt.red):
        m = QtWidgets.QGraphicsEllipseItem(-6, -6, 12, 12)
        m.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        m.setPen(QtGui.QPen(QtGui.QColor(color), 2))
        m.setPos(x, y)
        for a, b, c, d in [(-12, 0, 12, 0), (0, -12, 0, 12)]:
            ln = QtWidgets.QGraphicsLineItem(a, b, c, d, m)
            ln.setPen(QtGui.QPen(QtGui.QColor(color), 1))
        t = QtWidgets.QGraphicsSimpleTextItem(text, m)
        f = QtGui.QFont("Consolas", 10)
        f.setBold(True)
        t.setFont(f)
        t.setBrush(QtGui.QBrush(Qt.yellow))
        t.setPos(10, 8)
        bg = QtWidgets.QGraphicsRectItem(t.boundingRect(), t)
        bg.setBrush(QtGui.QColor(0, 0, 0, 170))
        bg.setPen(QtGui.QPen(Qt.NoPen))
        bg.setFlag(QtWidgets.QGraphicsItem.ItemStacksBehindParent)
        m.setZValue(10)
        self.scene().addItem(m)
        self.items_extra.append(m)

    def set_polygon(self, pts, color=Qt.green):
        poly = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in pts])
        it = self.scene().addPolygon(poly)
        pen = QtGui.QPen(QtGui.QColor(color), 3)
        pen.setCosmetic(True)
        it.setPen(pen)
        self.items_extra.append(it)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.pix is not None:
            p = self.mapToScene(e.pos())
            if self.pix.boundingRect().contains(p):
                self.clicked.emit(p.x(), p.y())
        super().mousePressEvent(e)

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automatic Georeferencing System (AGS)")
        self.resize(1500, 850)
        self.ref = None
        self.aerial_paths = []
        self.aerial_path = None
        self.aerial_orig = None   # RGB as loaded
        self.aerial = None        # RGB after rotation (what the user clicks on)
        self.rot_k = 0
        self.cache = {}           # (path, method, rot) -> match result

        # ---- controls
        self.btn_ref = QtWidgets.QPushButton("Load reference...")
        self.btn_dir = QtWidgets.QPushButton("Load aerial folder...")
        self.cmb_img = QtWidgets.QComboBox()
        self.cmb_img.setMinimumWidth(180)
        self.cmb_method = QtWidgets.QComboBox()
        self.cmb_method.addItems(available_methods())
        self.cmb_rot = QtWidgets.QComboBox()
        self.cmb_rot.addItems(["0", "90", "180", "270", "Auto"])
        self.btn_match = QtWidgets.QPushButton("Match now")
        self.btn_vis = QtWidgets.QPushButton("Show matches")
        self.btn_clear = QtWidgets.QPushButton("Clear marks")
        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_ref, self.btn_dir, QtWidgets.QLabel("Aerial:"), self.cmb_img,
                  QtWidgets.QLabel("Method:"), self.cmb_method,
                  QtWidgets.QLabel("Rotate (deg CCW):"), self.cmb_rot,
                  self.btn_match, self.btn_vis, self.btn_clear):
            top.addWidget(w)
        top.addStretch(1)

        self.view_a = ImageView("Aerial image")
        self.view_r = ImageView("Reference image")
        split = QtWidgets.QSplitter()
        split.addWidget(self.view_a)
        split.addWidget(self.view_r)
        split.setSizes([700, 700])

        self.lbl_result = QtWidgets.QLabel("Load the reference and an aerial image, then click a pixel.")
        self.lbl_result.setStyleSheet("font: 11pt 'Consolas'; padding: 4px;")

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addWidget(self.lbl_result)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        # ---- signals
        self.btn_ref.clicked.connect(self.choose_reference)
        self.btn_dir.clicked.connect(self.choose_folder)
        self.cmb_img.currentIndexChanged.connect(self.on_image_changed)
        self.cmb_rot.currentIndexChanged.connect(self.on_rotation_changed)
        self.cmb_method.currentIndexChanged.connect(lambda _: self.reset_marks())
        self.btn_match.clicked.connect(self.match_now)
        self.btn_vis.clicked.connect(self.show_matches)
        self.btn_clear.clicked.connect(self.reset_marks)
        self.view_a.clicked.connect(self.on_click)

    # ------------------------------------------------------------------ loading (step 1)
    def busy(self, msg):
        self.statusBar().showMessage(msg)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

    def idle(self, msg="Ready"):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(msg)

    def choose_reference(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reference image", HERE, "GeoTIFF (*.tif *.tiff)")
        if p:
            self.load_reference(p)

    def load_reference(self, path):
        self.busy("Loading reference...")
        try:
            self.ref = GeoReference(path)
        except Exception as ex:
            self.idle()
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open reference:\n{}".format(ex))
            return
        self.view_r.set_image(self.ref.rgb)
        self.cache.clear()
        self.idle("Reference loaded: {}x{} px".format(self.ref.w, self.ref.h))

    def choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with aerial images", HERE)
        if d:
            self.load_folder(d)

    def load_folder(self, d):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(d, e))
        files = [f for f in files if os.path.basename(f).lower() not in ("reference.tif", "reference.tiff")]
        nat = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]
        self.aerial_paths = sorted(files, key=nat)
        self.cmb_img.blockSignals(True)
        self.cmb_img.clear()
        self.cmb_img.addItems([os.path.basename(f) for f in self.aerial_paths])
        self.cmb_img.blockSignals(False)
        if self.aerial_paths:
            self.on_image_changed(0)
        self.statusBar().showMessage("{} aerial images loaded".format(len(self.aerial_paths)))

    def on_image_changed(self, i):
        if i < 0 or i >= len(self.aerial_paths):
            return
        self.aerial_path = self.aerial_paths[i]
        bgr = cv2.imread(self.aerial_path, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot read " + self.aerial_path)
            return
        self.aerial_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.cmb_rot.blockSignals(True)
        self.cmb_rot.setCurrentIndex(0)
        self.cmb_rot.blockSignals(False)
        self.rot_k = 0
        self.refresh_aerial()

    def on_rotation_changed(self, i):
        if self.aerial_orig is None:
            return
        if i < 4:
            self.rot_k = i
            self.refresh_aerial()
        else:
            self.reset_marks()

    def refresh_aerial(self):
        self.aerial = np.ascontiguousarray(np.rot90(self.aerial_orig, self.rot_k))
        self.view_a.set_image(self.aerial)
        self.view_r.clear_markers()
        self.lbl_result.setText("{}  (rotation {} deg)  - click a pixel".format(
            os.path.basename(self.aerial_path), 90 * self.rot_k))

    def reset_marks(self):
        self.view_a.clear_markers()
        self.view_r.clear_markers()

    # ------------------------------------------------------------------ steps 3-4
    def match_for_rot(self, k):
        method = self.cmb_method.currentText()
        key = (self.aerial_path, method, k)
        if key in self.cache:
            return self.cache[key]
        img = np.ascontiguousarray(np.rot90(self.aerial_orig, k))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t0 = time.time()
        pr, dr = reference_features(self.ref.path, self.ref.gray, method)
        pa, da = extract_features(gray, method, AERIAL_MAX_DIM)
        res = estimate_homography(pa, da, pr, dr, method)
        if res is not None:
            res["time"] = time.time() - t0
            res["k"] = k
            res["n_kp"] = (len(pa), len(pr))
        self.cache[key] = res
        return res

    def run_match(self):
        """Honours the rotation combo; 'Auto' tries 0/90/180/270 and keeps the best."""
        if self.ref is None or self.aerial_orig is None:
            QtWidgets.QMessageBox.information(self, "AGS", "Load the reference and an aerial image first.")
            return None
        self.busy("Extracting features and matching ({})...".format(self.cmb_method.currentText()))
        try:
            if self.cmb_rot.currentIndex() == 4:
                best = None
                for k in range(4):          # keep 0 deg if it already works, else take best of 90/180/270
                    r = self.match_for_rot(k)
                    if r is not None and (best is None or r["n_inl"] > best["n_inl"]):
                        best = r
                    if k == 0 and best is not None and best["n_inl"] >= 3 * MIN_INLIERS:
                        break
                if best is not None and best["k"] != self.rot_k:
                    self.rot_k = best["k"]
                    self.refresh_aerial()
                res = best
            else:
                res = self.match_for_rot(self.rot_k)
        finally:
            self.idle()
        if res is None or res["n_inl"] < MIN_INLIERS:
            n = 0 if res is None else res["n_inl"]
            self.statusBar().showMessage("Match failed/unreliable ({} inliers). Try another method or rotation.".format(n))
            return None
        self.statusBar().showMessage(
            "{}: {} good matches, {} RANSAC inliers, {:.1f}s".format(
                self.cmb_method.currentText(), res["n_good"], res["n_inl"], res["time"]))
        h, w = self.aerial.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        foot = cv2.perspectiveTransform(corners, res["H"]).reshape(-1, 2)
        self.view_r.clear_markers()
        self.view_r.set_polygon(foot)
        return res

    def match_now(self):
        self.run_match()

    def current_result(self):
        method = self.cmb_method.currentText()
        if self.cmb_rot.currentIndex() < 4:
            r = self.cache.get((self.aerial_path, method, self.rot_k))
            if r is not None and r["n_inl"] >= MIN_INLIERS:
                return r
        return self.run_match()

    # ------------------------------------------------------------------ steps 2, 5, 6, 7
    def on_click(self, x, y):
        if self.ref is None:
            QtWidgets.QMessageBox.information(self, "AGS", "Load the reference image first.")
            return
        res = self.current_result()           # step 3-4 (cached after first time)
        self.view_a.clear_markers()
        if res is None:
            self.view_a.add_marker(x, y, "({:.0f}, {:.0f})  no match".format(x, y), Qt.cyan)
            return
        pt = cv2.perspectiveTransform(np.float32([[[x, y]]]), res["H"])[0, 0]   # step 5
        rx, ry = float(pt[0]), float(pt[1])
        lat, lon = self.ref.pixel_to_latlon(rx, ry)                              # step 6
        inside = 0 <= rx < self.ref.w and 0 <= ry < self.ref.h
        txt = "px({:.0f},{:.0f})\nLat {:.6f}\nLon {:.6f}".format(x, y, lat, lon)  # step 7
        self.view_a.add_marker(x, y, txt)
        self.view_r.clear_markers()
        self.view_r.set_polygon(self._footprint(res))
        self.view_r.add_marker(rx, ry, "ref({:.0f},{:.0f})\nLat {:.6f}\nLon {:.6f}".format(rx, ry, lat, lon), Qt.cyan)
        self.lbl_result.setText(
            "{} | {} | clicked ({:.1f}, {:.1f}) -> reference ({:.1f}, {:.1f}) -> Lat {:.6f}, Lon {:.6f}{}".format(
                os.path.basename(self.aerial_path), self.cmb_method.currentText(), x, y, rx, ry, lat, lon,
                "" if inside else "  [outside reference]"))
        self.log_click(x, y, rx, ry, lat, lon, res)

    def _footprint(self, res):
        h, w = self.aerial.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(c, res["H"]).reshape(-1, 2)

    def log_click(self, x, y, rx, ry, lat, lon, res):
        new = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["image", "method", "rotation", "x", "y", "ref_x", "ref_y", "lat", "lon", "inliers"])
            wr.writerow([os.path.basename(self.aerial_path), self.cmb_method.currentText(), 90 * self.rot_k,
                         round(x, 1), round(y, 1), round(rx, 1), round(ry, 1),
                         "{:.7f}".format(lat), "{:.7f}".format(lon), res["n_inl"]])

    def show_matches(self):
        res = self.current_result()
        if res is None:
            return
        vis = draw_matches(self.aerial, self.ref.rgb, res["src"], res["dst"])
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("RANSAC inlier matches ({} inliers)".format(res["n_inl"]))
        lab = QtWidgets.QLabel()
        lab.setPixmap(to_pixmap(vis))
        sc = QtWidgets.QScrollArea()
        sc.setWidget(lab)
        QtWidgets.QVBoxLayout(dlg).addWidget(sc)
        dlg.resize(1400, 800)
        dlg.show()

    # ------------------------------------------------------------------ auto-load
    def autoload(self):
        for d in (HERE, os.path.join(HERE, "data"), os.getcwd()):
            ref = os.path.join(d, "reference.tif")
            if os.path.exists(ref):
                self.load_reference(ref)
                self.load_folder(d)
                return


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    w.autoload()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()