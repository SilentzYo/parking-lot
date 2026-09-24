import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLES = ["car", "motorcycle", "bus", "truck"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GREEN = (80, 200, 60)
RED = (60, 60, 230)


def load_spots(path):
    with open(path) as f:
        data = json.load(f)
    spots = {}
    for s in data["spots"]:
        spots[s["id"]] = np.array(s["points"], dtype=np.float32)
    return spots, data["image_size"]


def save_spots(path, spots, size):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    # one spot per line, otherwise json.dump makes the file huge
    lines = []
    for name, pts in spots.items():
        pts = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
        lines.append(json.dumps({"id": name, "points": pts}))
    with open(path, "w") as f:
        f.write('{\n "image_size": %s,\n "spots": [\n  ' % json.dumps(list(size)))
        f.write(",\n  ".join(lines))
        f.write("\n ]\n}\n")


def scale_spots(spots, size, width, height):
    sx = width / size[0]
    sy = height / size[1]
    return {name: pts * np.float32([sx, sy]) for name, pts in spots.items()}


def split_row(corners, n):
    # use a perspective transform so spots further away come out smaller
    src = np.float32([[0, 0], [n, 0], [n, 1], [0, 1]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners))
    top = np.float32([[i, 0] for i in range(n + 1)])
    bottom = np.float32([[i, 1] for i in range(n + 1)])
    top = cv2.perspectiveTransform(top[None], m)[0]
    bottom = cv2.perspectiveTransform(bottom[None], m)[0]
    spots = []
    for i in range(n):
        spots.append(np.float32([top[i], top[i + 1], bottom[i + 1], bottom[i]]))
    return spots


class Detector:
    def __init__(self, model_path, conf=0.25, imgsz=2560, tile=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.tile = tile
        self.classes = [i for i, name in self.model.names.items() if name in VEHICLES]

    def predict(self, images):
        boxes = []
        for i in range(0, len(images), 8):
            results = self.model.predict(images[i:i + 8], imgsz=self.imgsz, conf=self.conf,
                                         classes=self.classes, verbose=False)
            for r in results:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                boxes.append(np.column_stack([xyxy, conf]))
        return boxes

    def detect(self, frame):
        boxes = self.predict([frame])
        if self.tile:
            boxes += self.detect_tiles(frame)
        return merge_boxes(np.concatenate(boxes))

    def detect_tiles(self, frame):
        h, w = frame.shape[:2]
        t = self.tile
        positions = [(x, y) for y in tile_starts(h, t) for x in tile_starts(w, t)]
        crops = [frame[y:y + t, x:x + t] for x, y in positions]
        out = []
        for (x, y), crop, b in zip(positions, crops, self.predict(crops)):
            ch, cw = crop.shape[:2]
            # cars cut off by the tile edge show up whole in the next tile, so drop them here
            keep = np.ones(len(b), bool)
            if x > 0:
                keep &= b[:, 0] >= 2
            if y > 0:
                keep &= b[:, 1] >= 2
            if x + cw < w:
                keep &= b[:, 2] <= cw - 2
            if y + ch < h:
                keep &= b[:, 3] <= ch - 2
            b = b[keep]
            b[:, :4] += [x, y, x, y]
            out.append(b)
        return out


def tile_starts(size, tile):
    step = int(tile * 0.65)
    starts = list(range(0, max(size - tile, 0) + 1, step))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def merge_boxes(boxes, thresh=0.6):
    # basically nms, but overlap is measured against the smaller box so a
    # half-car box sitting inside a full one gets removed too
    if len(boxes) == 0:
        return boxes
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    removed = np.zeros(len(boxes), bool)
    for i in range(len(boxes)):
        if removed[i]:
            continue
        keep.append(i)
        w = np.minimum(boxes[i, 2], boxes[:, 2]) - np.maximum(boxes[i, 0], boxes[:, 0])
        h = np.minimum(boxes[i, 3], boxes[:, 3]) - np.maximum(boxes[i, 1], boxes[:, 1])
        overlap = w.clip(0) * h.clip(0) / np.minimum(areas[i], areas)
        removed |= overlap >= thresh
    return boxes[keep]


def find_taken(spots, boxes, min_overlap=0.2):
    # a car only counts for the spot it overlaps most. with an angled camera
    # the box usually spills into the next spot over
    taken = set()
    for x1, y1, x2, y2, _ in boxes:
        box = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        best = None
        best_area = 0
        for name, pts in spots.items():
            area, _ = cv2.intersectConvexConvex(pts, box)
            if area > best_area:
                best = name
                best_area = area
        if best and best_area >= min_overlap * cv2.contourArea(spots[best]):
            taken.add(best)
    return taken


def draw(frame, spots, taken, boxes=None):
    s = max(frame.shape[:2]) / 1300
    overlay = frame.copy()
    for name, pts in spots.items():
        cv2.fillPoly(overlay, [pts.astype(np.int32)], RED if name in taken else GREEN)
    out = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    if boxes is not None:
        for x1, y1, x2, y2, _ in boxes.astype(int):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for name, pts in spots.items():
        color = RED if name in taken else GREEN
        cv2.polylines(out, [pts.astype(np.int32)], True, color, max(1, round(s)))
        if name not in taken:
            cx, cy = pts.mean(axis=0).astype(int)
            put_text(out, name, cx, cy, 0.35 * s, center=True)

    free = len(spots) - len(taken)
    put_text(out, f"{free} of {len(spots)} spots free", int(12 * s), int(36 * s), 0.9 * s, pad=int(8 * s))
    return out


def put_text(img, text, x, y, scale, center=False, pad=2):
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if center:
        x -= tw // 2
        y += th // 2
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (30, 30, 30), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
