import argparse
import json
import os
import sys
import time

import cv2

from parking import IMAGE_EXTS, Detector, draw, find_taken, load_spots, scale_spots

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="image, video, webcam number or stream url")
parser.add_argument("--spots", required=True, help="json file from mark_spots.py")
parser.add_argument("--model", default="models/yolo11m.pt")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--imgsz", type=int, default=2560, help="bigger finds more far away cars but is slower")
parser.add_argument("--tile", type=int, default=0,
                    help="tile size, only needed if the cars are tiny (try 224 with --imgsz 896)")
parser.add_argument("--min-overlap", type=float, default=0.2, help="how much of a spot a car has to cover")
parser.add_argument("--stride", type=int, default=1, help="only check every nth frame")
parser.add_argument("--save", help="save the result image/video")
parser.add_argument("--json", help="write the status of each spot to this file")
parser.add_argument("--show", action="store_true")
parser.add_argument("--boxes", action="store_true", help="draw the detected cars too")
args = parser.parse_args()


def make_folder_for(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def print_status(spots, taken, frame_no=None):
    free = [name for name in spots if name not in taken]
    msg = f"{len(free)}/{len(spots)} free: {' '.join(free)}"
    if frame_no is not None:
        msg = f"[frame {frame_no}] {msg}"
    print(msg)

    if args.json:
        status = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "free": len(free),
            "total": len(spots),
            "spots": {name: "occupied" if name in taken else "free" for name in spots},
        }
        make_folder_for(args.json)
        with open(args.json, "w") as f:
            json.dump(status, f, indent=1)


spots, size = load_spots(args.spots)
detector = Detector(args.model, args.conf, args.imgsz, args.tile)

if args.source.lower().endswith(IMAGE_EXTS):
    frame = cv2.imread(args.source)
    if frame is None:
        print("couldn't read", args.source)
        sys.exit(1)
    h, w = frame.shape[:2]
    spots = scale_spots(spots, size, w, h)

    boxes = detector.detect(frame)
    taken = find_taken(spots, boxes, args.min_overlap)
    print_status(spots, taken)

    out = draw(frame, spots, taken, boxes if args.boxes else None)
    if args.save:
        make_folder_for(args.save)
        cv2.imwrite(args.save, out)
    if args.show:
        cv2.imshow("parking", out)
        cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("couldn't open", args.source)
        sys.exit(1)

    writer = None
    boxes = None
    taken = set()
    score = {}
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no == 0:
            h, w = frame.shape[:2]
            spots = scale_spots(spots, size, w, h)

        if frame_no % args.stride == 0:
            boxes = detector.detect(frame)
            now = find_taken(spots, boxes, args.min_overlap)
            # average over the last few checks so one missed car (or someone walking by) doesn't flip a spot
            for name in spots:
                x = 1.0 if name in now else 0.0
                score[name] = 0.3 * x + 0.7 * score.get(name, x)
            taken = {name for name in spots if score[name] >= 0.5}
            print_status(spots, taken, frame_no)

        out = draw(frame, spots, taken, boxes if args.boxes else None)
        if args.save:
            if writer is None:
                make_folder_for(args.save)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
        if args.show:
            cv2.imshow("parking", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
        frame_no += 1

    cap.release()
    if writer:
        writer.release()

cv2.destroyAllWindows()
