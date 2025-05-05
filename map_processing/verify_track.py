#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

def verify_track(image_path, outer_csv, inner_csv, center_csv):
    # 1) Load image (as RGB)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) Load the boundary & centerline points
    outer = np.loadtxt(outer_csv, delimiter=',')
    inner = np.loadtxt(inner_csv, delimiter=',')
    center = np.loadtxt(center_csv, delimiter=',')

    # 3) Plot
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img)
    ax.plot(outer[:,0], outer[:,1], '-r', linewidth=2, label='Outer Boundary')
    ax.plot(inner[:,0], inner[:,1], '-b', linewidth=2, label='Inner Boundary')
    ax.plot(center[:,0], center[:,1], '-g', linewidth=2, label='Centerline')

    ax.set_title('Track Extraction Verification')
    ax.axis('off')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Load track image + CSVs and overlay boundaries + centerline for verification."
    )
    p.add_argument('--image',  help="Path to the track image (PNG)", 
                   default='Silverstone_map_black.png')
    p.add_argument('--outer',  help="CSV of outer boundary vertices", 
                   default='outer_vertices_ss.csv')
    p.add_argument('--inner',  help="CSV of inner boundary vertices", 
                   default='inner_vertices_ss.csv')
    p.add_argument('--center', help="CSV of smoothed centerline", 
                   default='centerline_ss.csv')
    args = p.parse_args()

    verify_track(args.image, args.outer, args.inner, args.center)
