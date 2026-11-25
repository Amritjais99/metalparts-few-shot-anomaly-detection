"""Pinned, size-gated, resumable CPU-only download of MetalParts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
os.environ['HF_HOME'] = str(Path(__file__).resolve().parents[1] / 'data/raw/.cache/huggingface-home')
os.environ['HF_XET_CACHE'] = str(Path(__file__).resolve().parents[1] / 'data/raw/.cache/xet')
from huggingface_hub import snapshot_download
from huggingface_hub import constants
import httpx2
# The installed Hub streams large chunks by default. Small writes preserve
# partial-file progress when this connection drops mid-JPEG. Hash checks below
# remain authoritative; this changes transport buffering, not dataset bytes.
constants.DOWNLOAD_CHUNK_SIZE = 256 * 1024

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
REV = '82c642b2d18be60f2dab104f8dd4dea1f4ccfacb'

def main():
    started = time.time()
    listing = json.loads((ROOT/'common/configs/p1-dataset-listing.json').read_text())
    assert listing['revision'] == REV
    assert 'license: cc-by-4.0' in listing['card']
    files = [f for f in listing['files'] if f['rfilename'].startswith('MetalParts/')]
    assert files, 'No MetalParts files in the pinned listing'
    size = sum(f['size'] for f in files)
    free = shutil.disk_usage(ROOT).free
    assert size <= free - 10_000_000_000, (size, free)
    RAW.mkdir(parents=True, exist_ok=True)
    # Reuse the existing lossless MetalParts archive only when each member matches
    # the pinned listing's content hash. No extra archive/dataset is downloaded.
    by_path = {f['rfilename']: f for f in files}
    reused = 0
    archive = ROOT/'data/MetalParts.tar'
    if archive.exists():
        with tarfile.open(archive) as tar:
            for member in tar:
                name = member.name.removeprefix('./')
                if not member.isfile() or name not in by_path:
                    continue
                item = by_path[name]
                expected = item.get('lfs', {}).get('sha256')
                dest = RAW/name
                if dest.exists():
                    continue
                content = tar.extractfile(member).read()
                actual = hashlib.sha256(content).hexdigest() if expected else hashlib.sha1(
                    f'blob {len(content)}\0'.encode()+content).hexdigest()
                if actual != (expected or item['blobId']):
                    raise ValueError('Archive differs from pinned revision: '+name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
                reused += 1
        print(f'Existing archive reused: {reused} pinned files', flush=True)
    print(f'Download gate PASS: {len(files)} files, {size} bytes, free {free}', flush=True)
    for attempt in range(1,6):
        try:
            snapshot_download(repo_id='AmazonScience/RobustAD', repo_type='dataset', revision=REV,
                              allow_patterns=['MetalParts/**', 'README.md'],
                              local_dir=RAW, max_workers=16)
            break
        except httpx2.TransportError as error:
            print(f'Network interruption {attempt}/5: {type(error).__name__}; retaining partial files',flush=True)
            if attempt==5:
                raise
            time.sleep(5)
    errors = []
    for item in files:
        path = RAW/item['rfilename']
        if not path.exists() or path.stat().st_size != item['size']:
            errors.append(item['rfilename']); continue
        with path.open('rb') as f:
            if item.get('lfs'):
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
                expected = item['lfs']['sha256']
            else:
                h = hashlib.sha1(f'blob {item["size"]}\0'.encode())
                h.update(f.read()); digest = h.hexdigest(); expected = item['blobId']
        if digest != expected:
            errors.append(item['rfilename'])
    actual_paths={p.relative_to(RAW).as_posix()
                  for p in (RAW/'MetalParts').rglob('*') if p.is_file()}
    assert actual_paths == set(by_path), 'Unexpected or missing files in selected group folders'
    assert (RAW/'README.md').read_text(encoding='utf-8') == listing['card']
    result = dict(revision=REV, expected_files=len(files), expected_bytes=size,
                  verified_files=len(files)-len(errors), errors=errors, reused_archive_files=reused,
                  actual_group_file_count=len(actual_paths),
                  readme_sha256=hashlib.sha256((RAW/'README.md').read_bytes()).hexdigest(),
                  elapsed_seconds=time.time()-started, free_before=free,
                  free_after=shutil.disk_usage(ROOT).free, license='CC BY 4.0',
                  card_url=f'https://huggingface.co/datasets/AmazonScience/RobustAD/blob/{REV}/README.md')
    # Keep the completed P1 acquisition record as historical evidence.
    print(json.dumps(result),flush=True)
    assert not errors

if __name__ == '__main__':
    main()
