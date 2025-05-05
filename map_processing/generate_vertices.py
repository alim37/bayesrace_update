#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

# ----------------------------
# 1) Load & preprocess image
# ----------------------------
img = cv2.imread('Silverstone_map_black.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Couldn't open 'Silverstone_map_black.png'")

_, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# --------------------------------
# 2) Extract outer & inner contour
# --------------------------------
contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
contours = [c.reshape(-1,2) for c in contours if len(c) > 100]
contours.sort(key=lambda c: np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)),
              reverse=True)
outer, inner = contours[0], contours[1]

# ------------------------------------
# Helper: uniformly resample a contour
# ------------------------------------
def resample_contour(c, N=800):
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    us = np.linspace(0, L, N, endpoint=False)
    out = np.zeros((N,2), float)
    for i, u in enumerate(us):
        idx = np.searchsorted(s, u)
        t = (u - s[idx-1]) / (s[idx] - s[idx-1])
        out[i] = c[idx-1] + t * (c[idx] - c[idx-1])
    return out

outer_rs = resample_contour(outer, 800)
inner_rs = resample_contour(inner, 800)

# -----------------------------------------------------------------------------
# 3) Build centerline by pairing each outer point with its nearest inner point
# -----------------------------------------------------------------------------
# 3.1 build KD–Tree on inner boundary
tree = cKDTree(inner_rs)
# 3.2 for each outer point, find closest inner point
_, idxs = tree.query(outer_rs, k=1)
# 3.3 compute midpoints
mids = (outer_rs + inner_rs[idxs]) * 0.5

# 3.4 smooth with a closed spline (optional but usually helpful)
tck, _ = splprep(mids.T, s=1.0, per=True)
M = 1200
u2 = np.linspace(0, 1, M, endpoint=False)
x2, y2 = splev(u2, tck)
centerline = np.vstack([x2, y2]).T

# -----------------------
# 4) Save your three CSVs
# -----------------------
np.savetxt('outer_vertices_ss.csv', outer_rs,    fmt='%.3f', delimiter=',')
np.savetxt('inner_vertices_ss.csv', inner_rs,    fmt='%.3f', delimiter=',')
np.savetxt('centerline_ss.csv',      centerline, fmt='%.3f', delimiter=',')

print("Done! Generated outer_vertices.csv, inner_vertices.csv, centerline.csv")
