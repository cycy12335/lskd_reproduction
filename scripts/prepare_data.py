from pathlib import Path
import hashlib, json, tarfile, urllib.request, shutil
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
url = 'https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar'
archive = root / '.tools/cifar_teachers.tar'
if not archive.exists():
    partial = archive.with_suffix('.tar.part')
    print('Downloading official teacher archive', flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(archive)
with tarfile.open(archive) as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('resnet32x4_vanilla/ckpt_epoch_240.pth') and m.isfile()]
    if len(members) != 1:
        raise RuntimeError(f'Unexpected teacher archive members: {members}')
    member = members[0]
    target = root / 'models/pretrained/teacher/teacher_resnet32x4_cifar100.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tar.extractfile(member) as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
info = {
    'source': url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
    'model': 'resnet32x4', 'expected_accuracy_percent': 79.42,
    'expected_accuracy_source': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.pdf',
    'archive_member': member.name, 'local_filename': target.name,
    'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'checksum_note': 'Computed locally for provenance; not an independently published digest.',
}
(root / 'models/pretrained/teacher/source.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info, indent=2), flush=True)
from torchvision.datasets import CIFAR100
for train in (True, False):
    dataset = CIFAR100(root=root / 'data/raw', train=train, download=True)
    print(f'CIFAR-100 train={train}: {len(dataset)} samples, integrity passed', flush=True)
