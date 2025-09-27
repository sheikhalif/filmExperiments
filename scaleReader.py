#!/usr/bin/env python3
"""
Annotate spring balance video with force reading.
- Purple (top) -> use bottom edge (lowest y of its box) = 0 N
- Purple (bottom) -> use top edge (highest y of its box) = 2.5 N
- Green -> use center of its box
- Scale: 0..2.5 N linearly from top-mark to bottom-mark
"""

import cv2
import numpy as np
import argparse
from collections import deque

# ---- Default HSV ranges (from your project memory) ----
PURPLE_LOWER = np.array([5, 140, 140]) #Changed to orange
PURPLE_UPPER = np.array([20, 255, 255])  #Changed to orange

GREEN_LOWER  = np.array([80, 120, 120]) #Changed to teal
GREEN_UPPER  = np.array([95, 255, 255]) #Changed to teal

def get_boxes(mask, min_area=500):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h, a))
    return boxes

def mark_points_from_boxes(frame, top_box, bottom_box, green_box):
    # top mark: bottom edge midpoint
    tx, ty, tw, th, *_ = top_box
    top_mark = (int(tx + tw/2), int(ty + th))
    # bottom mark: top edge midpoint
    bx, by, bw, bh, *_ = bottom_box
    bot_mark = (int(bx + bw/2), int(by))
    # green center
    gx, gy, gw, gh, *_ = green_box
    g_center = (int(gx + gw/2), int(gy + gh/2))

    # Draw boxes
    cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 3)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 255), 3)
    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 3)

    # Draw marks
    cv2.circle(frame, top_mark, 10, (255, 0, 255), -1)
    cv2.circle(frame, bot_mark, 10, (255, 0, 255), -1)
    cv2.circle(frame, g_center, 10, (0, 255, 0), -1)

    return top_mark, bot_mark, g_center

def compute_reading(top_mark, bot_mark, g_center, max_N=2.5):
    # Guard against zero division
    span = float(bot_mark[1] - top_mark[1])
    if span <= 1: 
        return None, None
    frac = (g_center[1] - top_mark[1]) / span     # y grows downward
    frac = max(0.0, min(1.0, frac))
    reading = max_N * frac
    # projection y along the scale
    y_proj = int(top_mark[1] + frac * span)
    x_mid = int((top_mark[0] + bot_mark[0]) / 2)
    return reading, (x_mid, y_proj)

def annotate_frame(frame_bgr, max_N=2.5, smooth=None):
    """
    Returns (annotated_frame_bgr, reading_float or None)
    """
    frame = frame_bgr.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masks
    kernel = np.ones((9, 9), np.uint8)
    mask_p = cv2.inRange(hsv, PURPLE_LOWER, PURPLE_UPPER)
    mask_g = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Purple boxes: choose extreme top & bottom
    pboxes = get_boxes(mask_p, min_area=800)
    if len(pboxes) < 2:
        return frame, None  # not enough purple found
    pboxes.sort(key=lambda b: b[1])
    top_box, bottom_box = pboxes[0], pboxes[-1]

    # Green largest
    gboxes = get_boxes(mask_g, min_area=300)
    if len(gboxes) < 1:
        return frame, None
    gboxes.sort(key=lambda b: b[4], reverse=True)
    green_box = gboxes[0]

    # Marks & drawing
    top_mark, bot_mark, g_center = mark_points_from_boxes(frame, top_box, bottom_box, green_box)

    # Scale line
    cv2.line(frame, (top_mark[0], top_mark[1]), (bot_mark[0], bot_mark[1]), (0, 0, 0), 3)

    # Reading
    reading, proj = compute_reading(top_mark, bot_mark, g_center, max_N=max_N)
    if reading is not None:
        cv2.line(frame, (proj[0]-30, proj[1]), (proj[0]+30, proj[1]), (0, 0, 0), 4)
        cv2.putText(frame, f"{reading:.3f} N", (proj[0]+40, proj[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)

    return frame, reading

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Path to input video; omit to use webcam")
    ap.add_argument("--out",   type=str, default="spring_balance_annotated.mp4", help="Output video path")
    ap.add_argument("--maxN",  type=float, default=2.5, help="Top of scale (N)")
    ap.add_argument("--smooth", type=int, default=5, help="Median window for reading smoothing (frames). 0=off")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.video == "" else args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open input")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer (MP4 / H.264 fallback to mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Optional smoothing buffer
    buf = deque(maxlen=max(1, args.smooth))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, reading = annotate_frame(frame, max_N=args.maxN)

        # smoothing
        if args.smooth > 0 and reading is not None:
            buf.append(reading)
            reading_sm = np.median(buf)
            # overlay smoothed value at top-left
            cv2.putText(annotated, f"Smoothed: {reading_sm:.3f} N", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        elif args.smooth > 0:
            cv2.putText(annotated, "Smoothed: --", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)

        writer.write(annotated)

        # Optional live preview (press q to quit)
        cv2.imshow("Annotated", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
