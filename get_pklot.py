import argparse
import os
import tarfile
import urllib.request
import zlib

URL = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data")
parser.add_argument("--every", type=int, default=20, help="keep 1 out of every n photos (1 = all ~12k)")
args = parser.parse_args()

# the archive is 4.9gb and shuffled, so stream through all of it and only save
# some of the full photos + their xml labels (skipping the cropped PKLotSegmented stuff)
kept = 0
with urllib.request.urlopen(URL) as resp:
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith("PKLot/PKLot/"):
                continue
            # pick by filename so a photo and its xml always get picked together
            stem = os.path.splitext(os.path.basename(member.name))[0]
            if zlib.crc32(stem.encode()) % args.every != 0:
                continue
            tar.extract(member, args.out, filter="data")
            if member.name.endswith(".jpg"):
                kept += 1
                if kept % 50 == 0:
                    print(kept, "photos so far", flush=True)

print("done,", kept, "photos saved to", args.out)
