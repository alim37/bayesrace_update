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

def lateral_bounds_at_node(p, n, inner_line, outer_line, search_dist=5.0):
    ray = LineString([tuple(p - search_dist*n), tuple(p + search_dist*n)])
    i_inner = ray.intersection(inner_line)
    i_outer = ray.intersection(outer_line)

    def pick_point(geom):
        if geom.geom_type == 'Point':
            return np.array(geom.coords[0])
        if geom.geom_type.startswith('Multi') or isinstance(geom, GeometryCollection):
            pts = np.array([pt.coords[0] for pt in geom if pt.geom_type == 'Point'])
            dists = np.linalg.norm(pts - p, axis=1)
            return pts[np.argmin(dists)]
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            dists = np.linalg.norm(coords - p, axis=1)
            return coords[np.argmin(dists)]
        raise RuntimeError(f"Unexpected geometry {geom.geom_type!r}")

    if i_inner.is_empty or i_outer.is_empty:
        # fallback to nearest-point on each boundary
        pt = Point(p)
        pi = np.array(nearest_points(pt, inner_line)[1].coords[0])
        po = np.array(nearest_points(pt, outer_line)[1].coords[0])
    else:
        pi = pick_point(i_inner)
        po = pick_point(i_outer)

    di = np.dot(pi - p, n)
    do = np.dot(po - p, n)
    return sorted([di, do])

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
        p, n = center_pts[idx], normals[idx]
        low, high = lateral_bounds_at_node(p, n, inner_line, outer_line)
        low, high = low*scale, high*scale
        w = np.random.uniform(low, high)
        xs[i], ys[i] = p[0] + w*n[0], p[1] + w*n[1]

    t_nodes = np.linspace(0, 1, n_nodes)
    cs_x = CubicSpline(t_nodes, xs)
    cs_y = CubicSpline(t_nodes, ys)
    t_smooth = np.linspace(0, 1, n_spline)
    return cs_x(t_smooth), cs_y(t_smooth)

def estimate_lap_time(x, y, mu=1.0, g=9.81, vmax=np.inf):
    # first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    # second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # curvature
    kappa = np.abs(dx*ddy - dy*ddx) / (dx*dx + dy*dy)**1.5
    # max lateral speed
    v_lat = np.sqrt(mu * g / np.maximum(kappa, 1e-6))
    # enforce straight-line cap
    v_allowed = np.minimum(v_lat, vmax)
    # segment lengths
    ds = np.hypot(np.diff(x), np.diff(y))
    # time per segment
    t_seg = ds / v_allowed[:-1]
    return np.sum(t_seg)

if __name__ == "__main__":
    inner, outer, center = load_track()

    best_time = float('inf')
    best_line = None
    all_lines = []

    # iterations
    for i in range(50):
        x_line, y_line = generate_racing_line()
        t = estimate_lap_time(x_line, y_line)
        all_lines.append((x_line, y_line, t))
        if t < best_time:
            best_time = t
            best_line = (x_line, y_line)

    # plot
    plt.figure(figsize=(8,8))
    plt.plot(inner[:,0], inner[:,1], 'k--', label="Inner boundary")
    plt.plot(outer[:,0], outer[:,1], 'k--', label="Outer boundary")
    plt.plot(center[:,0], center[:,1], 'gray',  label="Centerline")

    # plot non-best lines with low opacity
    for x, y, t in all_lines:
        if (x, y) is best_line:
            continue
        plt.plot(x, y, color='blue', alpha=0.2, label="_nolegend_")

    # plot best line with red
    xb, yb = best_line
    plt.plot(xb, yb, color='red', linewidth=2,
             label=f"Best line (est. time = {best_time:.2f}s)")

    plt.axis('equal')
    plt.legend()
    plt.title("Best Racing Line Highlighted")
    plt.show()
