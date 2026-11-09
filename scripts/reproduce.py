#!/usr/bin/env python3
"""Verify supplied numerical inputs and reproduce the paper's closed analyses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPOSITORY)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--tex', action='store_true', help='Render publication figures with TeX fonts')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if root == out or root in out.parents:
        parser.error('Write reproduction outputs outside the code/results repository.')
    data_root = root / 'results'
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe numerical input path: ' + name)
        path = data_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing or linked numerical input: ' + name)
        if (path.stat().st_size != expected['bytes'] or
                checksum(path) != expected['sha256']):
            raise RuntimeError('Numerical input checksum mismatch: ' + name)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(out / 'mplconfig'))
    stages = []
    if not args.verify_only:
        for name, filename in [
            ('closed', 'reproduce_closed.py'),
            ('evidence', 'reproduce_evidence.py'),
            ('observable', 'reproduce_observable.py'),
            ('matched-case', 'matched_case.py'),
            ('figures', 'reproduce_figures.py'),
        ]:
            command = [sys.executable, '-B', str(root / 'scripts' / filename),
                       '--root', str(root), '--out', str(out / name)]
            if name == 'figures' and args.tex:
                command.append('--tex')
            start = time.monotonic()
            result = subprocess.run(command, cwd=out, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out / f'{name}.log').write_text(result.stdout)
            stages.append({'name': name, 'exit_code': result.returncode,
                           'seconds': time.monotonic() - start})
            if result.returncode:
                log_path = out / f'{name}.log'
                raise RuntimeError(f'{name} reproduction failed; see {log_path}\n'
                                   + result.stdout[-3000:])
    receipt = {
        'status': 'passed', 'numerical_files_checked': len(manifest['files']),
        'input_manifest_sha256': checksum(manifest_path),
        'replay': not args.verify_only, 'stages': stages,
        'inference': False, 'fitting': False,
    }
    (out / 'REPRODUCTION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
