"""Stage the public THINGS-EEG2 17-channel data and verify the OSF archives."""
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

OSF_LIST = "https://api.osf.io/v2/nodes/anp5v/files/osfstorage/?page[size]=50"
NEMAR_OBJECTS = "https://nemar.s3.us-east-2.amazonaws.com/nm000232/objects/"
ROOT = Path(__file__).resolve().parents[1] / "data" / "THINGS-EEG2"
ARCHIVES = ROOT / "archives"
PREPROCESSED = ROOT / "Preprocessed_data"
CHUNK_SIZE = 32 * 1024**2


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_storage(url, size):
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-0/{size}":
                    raise OSError("OSF range probe did not match the archive size")
                response.read(1)
                return response.url
        except (OSError, IncompleteRead):
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


def download_chunk(sources, size, start, end, part):
    for attempt in range(6):
        offset = part.stat().st_size if part.exists() else 0
        if offset == end - start:
            return part
        if offset > end - start:
            raise ValueError(f"Oversize partial file: {part}")
        interval = f"{start + offset}-{end - 1}"
        target = sources[attempt % len(sources)]
        request = Request(target, headers={"Range": f"bytes={interval}", "Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {interval}/{size}":
                    raise OSError("Unexpected HTTP range")
                with part.open("ab") as stream:
                    shutil.copyfileobj(response, stream, length=64 * 1024)
            if part.stat().st_size != end - start:
                raise OSError("Incomplete HTTP range")
            return part
        except (OSError, IncompleteRead):
            if attempt == 5:
                raise
            time.sleep(attempt + 1)


def stage_subject(record, chunks, old_sub02):
    attributes = record["attributes"]
    name = attributes["name"]
    size = attributes["size"]
    expected_hash = attributes["extra"]["hashes"]["sha256"]
    url = record["links"]["download"]
    mirror = f"{NEMAR_OBJECTS}SHA256E-s{size}--{expected_hash}.zip"
    archive = ARCHIVES / name
    temp = archive.with_suffix(".zip.part")
    part_dir = ARCHIVES / (archive.stem + ".parts")
    if not archive.exists():
        if name == "sub-02.zip" and old_sub02 is not None and old_sub02.is_file():
            shutil.copyfile(old_sub02, temp)
        else:
            part_dir.mkdir(exist_ok=True)
            storage = resolve_storage(url, size)
            jobs = []
            for index, start in enumerate(range(0, size, CHUNK_SIZE)):
                part = part_dir / f"{start:012d}.part"
                sources = (storage, mirror, url) if index % 2 == 0 else (mirror, storage, url)
                jobs.append(chunks.submit(download_chunk, sources, size, start, min(start + CHUNK_SIZE, size), part))
            with temp.open("wb") as stream:
                for job in jobs:
                    with job.result().open("rb") as piece:
                        shutil.copyfileobj(piece, stream, length=1024**2)
        if temp.stat().st_size != size or sha256(temp) != expected_hash:
            raise ValueError(f"OSF SHA-256 or size mismatch for {name}")
        temp.replace(archive)
    elif archive.stat().st_size != size or sha256(archive) != expected_hash:
        raise ValueError(f"Existing archive does not match OSF: {archive}")

    subject_dir = PREPROCESSED / archive.stem
    subject_dir.mkdir(exist_ok=True)
    with ZipFile(archive) as source:
        for split in ("training", "test"):
            filename = f"preprocessed_eeg_{split}.npy"
            member = source.getinfo(f"{archive.stem}/{filename}")
            output = subject_dir / filename
            pending = output.with_suffix(".npy.part")
            with source.open(member) as data, pending.open("wb") as stream:
                shutil.copyfileobj(data, stream, length=1024**2)
            with pending.open("rb") as stream:
                if stream.read(6) != b"\x93NUMPY" or pending.stat().st_size != member.file_size:
                    raise ValueError(f"Invalid EEG data in {name}: {filename}")
            pending.replace(output)
    if part_dir.is_dir():
        for start in range(0, size, CHUNK_SIZE):
            (part_dir / f"{start:012d}.part").unlink(missing_ok=True)
        part_dir.rmdir()
    print(f"Verified and extracted {name} ({size:,} bytes)", flush=True)
    return {"archive": name, "bytes": size, "sha256": expected_hash, "download": url, "mirror": mirror}


def main():
    old_sub02 = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    with urlopen(OSF_LIST, timeout=45) as response:
        records = {row["attributes"]["name"]: row for row in json.load(response)["data"]}
    targets = [records[f"sub-{subject:02d}.zip"] for subject in range(1, 11)]
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    print("Staging 10 subjects from OSF and its checksum-matched NEMAR mirror", flush=True)
    verified = {}
    with ThreadPoolExecutor(max_workers=40) as chunks, ThreadPoolExecutor(max_workers=4) as subjects:
        jobs = {subjects.submit(stage_subject, row, chunks, old_sub02): row["attributes"]["name"] for row in targets}
        for job in as_completed(jobs):
            name = jobs[job]
            try:
                verified[name] = job.result()
            except (OSError, IncompleteRead, ValueError) as error:
                print(f"{name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    manifest = [verified[name] for name in sorted(verified)]
    (ROOT / "osf_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if len(verified) != len(targets):
        raise SystemExit(f"Only {len(verified)}/{len(targets)} subjects verified. Rerun to resume partial files.")


if __name__ == "__main__":
    main()
