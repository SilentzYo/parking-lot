import argparse
import os
import string
import sys

import cv2
import numpy as np

from parking import IMAGE_EXTS, load_spots, put_text, save_spots, scale_spots, split_row

LETTERS = string.ascii_uppercase
ROW_NAMES = list(LETTERS) + [a + b for a in LETTERS for b in LETTERS]


class Editor:
    def __init__(self, image, spots, size, path):
        self.image = image
        self.spots = spots
        self.size = size
        self.path = path
        self.scale = min(1, 1600 / image.shape[1], 900 / image.shape[0])
        self.corners = []
        self.count = ""
        self.undo_stack = []
        self.saved = True
        self.msg = ""

    def mouse(self, event, x, y, flags, param):
        x = x / self.scale
        y = y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))
            if len(self.corners) == 4:
                self.fix_corners()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for name, pts in self.spots.items():
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    self.remember()
                    del self.spots[name]
                    self.msg = "deleted " + name
                    break

    def fix_corners(self):
        if not cv2.isContourConvex(np.float32(self.corners)):
            self.corners[2], self.corners[3] = self.corners[3], self.corners[2]
        if not cv2.isContourConvex(np.float32(self.corners)):
            self.corners = []
            self.msg = "that's not a 4 sided shape, try again"

    def num_spots(self):
        if self.count == "":
            return 1
        return max(1, int(self.count))

    def next_row(self):
        used = {name.rstrip("0123456789") for name in self.spots}
        for row in ROW_NAMES:
            if row not in used:
                return row

    def add_row(self):
        n = self.num_spots()
        row = self.next_row()
        self.remember()
        for i, pts in enumerate(split_row(self.corners, n)):
            self.spots[f"{row}{i + 1}"] = pts
        self.msg = f"added row {row} ({n} spots)"
        self.corners = []
        self.count = ""

    def remember(self):
        self.undo_stack.append(dict(self.spots))
        self.saved = False

    def undo(self):
        if self.corners:
            self.corners.pop()
            self.count = ""
        elif self.undo_stack:
            self.spots = self.undo_stack.pop()
            self.saved = False
            self.msg = "undo"

    def key(self, k):
        # returns False when it's time to quit
        if len(self.corners) == 4:
            if ord("0") <= k <= ord("9"):
                self.count += chr(k)
                return True
            if k == 8:  # backspace
                self.count = self.count[:-1]
                return True
            if k in (10, 13):  # enter
                self.add_row()
                return True

        if k == 27 and self.corners:  # esc cancels the shape you're drawing
            self.corners = []
            self.count = ""
        elif k == ord("u"):
            self.undo()
        elif k == ord("s"):
            save_spots(self.path, self.spots, self.size)
            self.saved = True
            self.msg = f"saved {len(self.spots)} spots"
        elif k == ord("q") or k == 27:
            if self.saved or self.msg.startswith("unsaved"):
                return False
            self.msg = "unsaved changes! s to save, q again to quit"
        return True

    def draw(self):
        img = self.image.copy()
        t = max(1, round(max(img.shape[:2]) / 1000))
        for name, pts in self.spots.items():
            cv2.polylines(img, [pts.astype(np.int32)], True, (0, 220, 255), t)
            cx, cy = pts.mean(axis=0).astype(int)
            put_text(img, name, cx, cy, 0.32 * t, center=True)

        if len(self.corners) == 4:
            for pts in split_row(self.corners, self.num_spots()):
                cv2.polylines(img, [pts.astype(np.int32)], True, (255, 200, 0), t)
        elif self.corners:
            pts = np.int32(self.corners)
            cv2.polylines(img, [pts], False, (255, 200, 0), t)
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 3 * t, (255, 200, 0), -1)

        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        lines = [
            "left click: add a corner. after 4 corners type how many spots are in the row, then enter",
            "right click: delete spot   u: undo   s: save   esc: cancel   q: quit",
        ]
        if len(self.corners) == 4:
            lines.append(f"spots in this row: {self.num_spots()}")
        else:
            lines.append(f"{len(self.spots)} spots   corners: {len(self.corners)}/4   {self.msg}")
        for i, line in enumerate(lines):
            put_text(img, line, 8, 22 + 22 * i, 0.5, pad=3)
        return img


def grab_frame(source):
    if source.lower().endswith(IMAGE_EXTS):
        return cv2.imread(source)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    frame = None
    # webcams are usually dark for the first few frames
    for _ in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="image, video, webcam number or stream url")
    parser.add_argument("--spots", required=True, help="json file to save to (gets loaded if it already exists)")
    args = parser.parse_args()

    image = grab_frame(args.source)
    if image is None:
        print("couldn't get an image from", args.source)
        sys.exit(1)
    h, w = image.shape[:2]
    spots = {}
    if os.path.exists(args.spots):
        spots, size = load_spots(args.spots)
        spots = scale_spots(spots, size, w, h)

    editor = Editor(image, spots, (w, h), args.spots)
    cv2.namedWindow("spots")
    cv2.setMouseCallback("spots", editor.mouse)
    while cv2.getWindowProperty("spots", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.imshow("spots", editor.draw())
        k = cv2.waitKey(30)
        if k != -1 and not editor.key(k & 0xFF):
            break
    cv2.destroyAllWindows()
