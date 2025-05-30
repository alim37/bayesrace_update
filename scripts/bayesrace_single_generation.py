import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, GeometryCollection
from shapely.ops import nearest_points
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def load_track():
    inner = pd.read_csv('inner_vertices.csv').to_numpy()
    outer = pd.read_csv('outer_vertices.csv').to_numpy()
    center = pd.read_csv('centerline.csv').to_numpy()
    return inner, outer, center

def compute_normals(center):
    dx = np.gradient(center[:,0])
    dy = np.gradient(center[:,1])
    lengths = np.hypot(dx, dy)
    nx = -dy / lengths
    ny =  dx / lengths
    return np.column_stack((nx, ny))

def lateral_bounds_at_node(p, n, inner_line, outer_line,
                           search_dist=5.0, max_fallback_dist=100.0):
    # 1) Try the normal‐ray intersection
    ray = LineString([tuple(p - search_dist*n), tuple(p + search_dist*n)])
    i_inner = ray.intersection(inner_line)
    i_outer = ray.intersection(outer_line)

    def pick_point(geom):
        # single point
        if geom.geom_type == 'Point':
            return np.array(geom.coords[0])
        # a collection of points
        if geom.geom_type.startswith('Multi') or isinstance(geom, GeometryCollection):
            pts = np.array([pt.coords[0] for pt in geom if pt.geom_type == 'Point'])
            dists = np.linalg.norm(pts - p, axis=1)
            return pts[np.argmin(dists)]
        # a segment
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            dists = np.linalg.norm(coords - p, axis=1)
            return coords[np.argmin(dists)]
        raise RuntimeError(f"Unexpected geometry {geom.geom_type!r}")

    # If either intersection is empty, fall back to nearest‐point
    if i_inner.is_empty or i_outer.is_empty:
        pt = Point(p)
        pi = np.array(nearest_points(pt, inner_line)[1].coords[0])
        po = np.array(nearest_points(pt, outer_line)[1].coords[0])
    else:
        pi = pick_point(i_inner)
        po = pick_point(i_outer)

    # Compute signed distances along the normal
    di = np.dot(pi - p, n)
    do = np.dot(po - p, n)
    low, high = sorted([di, do])
    return low, high


def generate_racing_line(n_nodes=20, scale=0.9, n_spline=200):
    inner_pts, outer_pts, center_pts = load_track()
    normals = compute_normals(center_pts)
    inner_line = LineString(inner_pts)
    outer_line = LineString(outer_pts)

    M = len(center_pts)
    idxs = np.linspace(0, M-1, n_nodes, dtype=int)

    xs = np.zeros(n_nodes)
    ys = np.zeros(n_nodes)

    for i, idx in enumerate(idxs):
        p = center_pts[idx]
        n = normals[idx]
        low, high = lateral_bounds_at_node(p, n, inner_line, outer_line)
        low *= scale
        high *= scale
        w = np.random.uniform(low, high)
        xs[i] = p[0] + w * n[0]
        ys[i] = p[1] + w * n[1]

    t_nodes = np.linspace(0, 1, n_nodes)
    cs_x = CubicSpline(t_nodes, xs)
    cs_y = CubicSpline(t_nodes, ys)
    t_smooth = np.linspace(0, 1, n_spline)
    x_line = cs_x(t_smooth)
    y_line = cs_y(t_smooth)

    return inner_pts, outer_pts, center_pts, x_line, y_line, xs, ys

if __name__ == "__main__":
    inner, outer, center, x_line, y_line, x_nodes, y_nodes = generate_racing_line()

    plt.figure(figsize=(8,8))
    plt.plot(inner[:,0], inner[:,1], 'k--', label="Inner boundary")
    plt.plot(outer[:,0], outer[:,1], 'k--', label="Outer boundary")
    plt.plot(center[:,0], center[:,1], 'gray',  label="Centerline")
    plt.plot(x_line, y_line,   'r-',   label="Racing line")
    plt.scatter(x_nodes, y_nodes, c='r', s=30, label="Sample nodes")
    plt.axis('equal')
    plt.legend()
    plt.title("Generated Racing Line")
    plt.show()
