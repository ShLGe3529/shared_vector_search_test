#!/usr/bin/env python3
"""Download the immutable public inputs, with resumable 10M byte-range crops."""

import argparse
import sys
import urllib.request
from pathlib import Path


FILES = {
    "sift10m": [
        (
            "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
            "data/sift10m/raw/base.10M.u8bin.part", 1_280_000_008, True,
        ),
        (
            "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
            "data/sift10m/raw/query.public.10K.u8bin", 1_280_008, False,
        ),
        (
            "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M",
            "data/sift10m/raw/groundtruth.public.10K.ibin", 8_000_008, False,
        ),
    ],
    "cohere10m": [
        (
            "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia_base.bin",
            "data/cohere10m/raw/base.10M.fbin.part", 30_720_000_008, True,
        ),
        (
            "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia_query.bin",
            "data/cohere10m/raw/wikipedia_query.bin", 15_360_008, False,
        ),
        (
            "https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M/wikipedia-10M",
            "data/cohere10m/raw/wikipedia-10M", 4_000_008, False,
        ),
    ],
}


def download(root: Path, url: str, relative: str, expected: int,
             range_crop: bool) -> None:
    path = root / relative
    final_path = Path(str(path).removesuffix(".part"))
    if range_crop and final_path.exists() and final_path.stat().st_size == expected:
        print(f"already present: {final_path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.stat().st_size if path.exists() else 0
    if current == expected:
        print(f"download complete: {path}", flush=True)
        return
    if current > expected:
        raise RuntimeError(f"oversized partial file: {path}")
    headers = {}
    if range_crop or current:
        headers["Range"] = f"bytes={current}-{expected - 1}"
    request = urllib.request.Request(url, headers=headers)
    print(f"downloading {url} -> {path} from byte {current}", flush=True)
    with urllib.request.urlopen(request) as response:
        if headers and response.status != 206:
            raise RuntimeError(f"server ignored byte range for {url}")
        mode = "ab" if current else "wb"
        with path.open(mode) as handle:
            while block := response.read(8 << 20):
                handle.write(block)
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"short download {path}: {path.stat().st_size} != {expected}"
        )
    print(f"download complete: {path} ({expected} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(FILES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("provide --dataset or --all")
    datasets = list(FILES) if args.all else [args.dataset]
    for dataset in datasets:
        for item in FILES[dataset]:
            download(args.root.resolve(), *item)


if __name__ == "__main__":
    main()

