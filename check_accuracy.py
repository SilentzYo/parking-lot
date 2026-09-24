import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np

from parking import Detector, draw, find_taken

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="?", default="data", help="folder with PKLot photos + xml files (from get_pklot.py)")
parser.add_argument("--model", default="models/yolo11m.pt")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--imgsz", type=int, default=2560)
parser.add_argument("--tile", type=int, default=0)
parser.add_argument("--min-overlap", type=float, default=0.2)
parser.add_argument("--limit", type=int, help="only check this many photos, spread out over all of them")
parser.add_argument("--mistakes", help="save photos with wrong spots to this folder")
args = parser.parse_args()


def read_labels(xml_path):
    spots = {}
    occupied = set()
    for space in ET.parse(xml_path).getroot().findall("space"):
        contour = space.find("contour")
        # a few spaces in pklot are missing the label or the outline
        if space.get("occupied") is None or contour is None:
            continue
        pts = np.float32([[float(p.get("x")), float(p.get("y"))] for p in contour.findall("point")])
        if len(pts) < 3:
            continue
        name = space.get("id")
        spots[name] = cv2.convexHull(pts).reshape(-1, 2)
        if space.get("occupied") == "1":
            occupied.add(name)
    return spots, occupied


photos = sorted(glob.glob(os.path.join(args.folder, "**", "*.jpg"), recursive=True))
photos = [p for p in photos if os.path.exists(p[:-4] + ".xml")]
if not photos:
    print("no labeled photos in", args.folder, "- run get_pklot.py first")
    sys.exit(1)
if args.limit and args.limit < len(photos):
    step = len(photos) / args.limit
    photos = [photos[int(i * step)] for i in range(args.limit)]

detector = Detector(args.model, args.conf, args.imgsz, args.tile)

# per group: photos, spots right, wrongly free (car there but we said free), wrongly taken
stats = defaultdict(lambda: [0, 0, 0, 0])
start = time.time()
for i, path in enumerate(photos):
    spots, occupied = read_labels(path[:-4] + ".xml")
    frame = cv2.imread(path)
    boxes = detector.detect(frame)
    taken = find_taken(spots, boxes, args.min_overlap)

    wrongly_free = occupied - taken
    wrongly_taken = taken - occupied
    right = len(spots) - len(wrongly_free) - len(wrongly_taken)

    parts = os.path.normpath(path).split(os.sep)
    lot, weather = parts[-4], parts[-3].lower()
    for group in ("all", lot, f"{lot} {weather}"):
        s = stats[group]
        s[0] += 1
        s[1] += right
        s[2] += len(wrongly_free)
        s[3] += len(wrongly_taken)

    if args.mistakes and (wrongly_free or wrongly_taken):
        out = draw(frame, spots, taken, boxes)
        for name in wrongly_free | wrongly_taken:
            cv2.polylines(out, [spots[name].astype(np.int32)], True, (0, 255, 255), 3)
        os.makedirs(args.mistakes, exist_ok=True)
        cv2.imwrite(os.path.join(args.mistakes, f"{lot}_{os.path.basename(path)}"), out)

    if (i + 1) % 25 == 0:
        print(f"{i + 1}/{len(photos)} photos checked", flush=True)

took = time.time() - start
print()
print(f"{'':16} {'photos':>6} {'spots':>7} {'correct':>8} {'wrongly free':>13} {'wrongly taken':>14}")
for group in sorted(stats):
    n, right, wf, wt = stats[group]
    total = right + wf + wt
    print(f"{group:16} {n:6} {total:7} {right / total:8.1%} {wf / total:13.1%} {wt / total:14.1%}")
print(f"\n{took / len(photos):.2f}s per photo")
