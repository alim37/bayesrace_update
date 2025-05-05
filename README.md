# Generating a racing line using Bayesian optimization

## Core Functionalities
- **`load_track()`**  
  - Reads three CSVs of map (inner boundary, outer boundary, centerline) into NumPy arrays  
- **`compute_normals(center)`**  
  - Computes unit normals at each centerline point via finite-difference on (x, y)  
- **`lateral_bounds_at_node(p, n, inner_line, outer_line)`**  
  - Casts short “normal” ray at point `p` in direction `n`  
  - Intersects with inner/outer `LineString` boundaries  
  - Falls back to the true nearest-point if intersection is empty  
  - Returns `[d_min, d_max]` lateral offsets along `n`  
- **`generate_racing_line(n_nodes=x, scale=0.9, n_spline=x)`**  
  - Samples `n_nodes` evenly spaced along the centerline  
  - For each node:  
    - Computes lateral bounds via `lateral_bounds_at_node()`  
    - Draws a random offset `w ∈ [d_min, d_max] * scale`  
    - Projects to `(x_i, y_i) = p_i + w ⋅ n_i`  
  - Fits cubic splines through `(x_i, y_i)` → smooth racing line `(x(t), y(t))`  
- **`estimate_lap_time(x, y, mu=1.0, g=9.81, vmax=∞)`**  
  - Approximates lap time by:  
    1. Computing curvature κ via first/second finite-differences  
    2. Computing local lateral-limit speed `v_lat = sqrt(mu⋅g / κ)` (capped by `vmax`)  
    3. Summing Δt = Δs / v_allowed over each segment
   
## Running
Process map and then change iteration parameters (number of samples along centerline and number of splines) for best fit.

## Testing different maps
This repo (https://github.com/f1tenth/f1tenth_racetracks) has all the F1 tracks in order to test each one.
