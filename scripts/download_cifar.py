"""Parallel range download of the official CIFAR-100 archive with torchvision MD5."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import math
import shutil
import urllib.request
import time
from torchvision.datasets import CIFAR100

root = Path(__file__).resolve().parents[1]
parts = root / '.tools/cifar100_parts'
parts.mkdir(parents=True, exist_ok=True)
total = 169001437
chunk_size = 1024 * 1024
url = CIFAR100.url


def download(index):
    start = index * chunk_size
    end = min(total, start + chunk_size) - 1
    target = parts / f'{index:03d}.part'
    if target.exists() and target.stat().st_size == end - start + 1:
        return index
    for attempt in range(4):
        try:
            request = urllib.request.Request(url + f'?range={start}-{end}', headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                    raise RuntimeError('Server did not return the requested range')
                with target.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=128 * 1024)
            if target.stat().st_size != end - start + 1:
                raise RuntimeError('Incomplete range response')
            return index
        except Exception as error:
            print(f"Range {index} retry {attempt}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


if __name__ == '__main__':
    count = math.ceil(total / chunk_size)
    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = [executor.submit(download, index) for index in range(count)]
        for done, future in enumerate(as_completed(jobs), 1):
            print(f'Completed range {future.result()}; {done}/{count}', flush=True)
    target = root / '.tools/cifar100_complete.tar.gz'
    digest = hashlib.md5()
    with target.open('wb') as stream:
        for index in range(count):
            with (parts / f'{index:03d}.part').open('rb') as src:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    digest.update(chunk)
                    stream.write(chunk)
    if digest.hexdigest() != CIFAR100.tgz_md5:
        raise RuntimeError(f'Archive integrity mismatch: {digest.hexdigest()}')
    print(f'Official CIFAR-100 archive verified: {digest.hexdigest()}', flush=True)
