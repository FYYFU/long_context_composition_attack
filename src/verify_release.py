#!/usr/bin/env python3
"""Check publishable files, archive schemas, and the historical paper score grid."""
from __future__ import annotations

import ast
import gzip
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'(?:sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
LOCAL_PATH = re.compile(r'/(?:data2?|home)/yfu093/')


def main() -> None:
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT,
    ).decode().split('\0')
    files = [ROOT / name for name in names if name]
    failures = []
    python_count = shell_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size >= 100 * 1024 * 1024:
            failures.append(f'File exceeds 100 MiB: {relative}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(relative))
            python_count += 1
        if path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            shell_count += 1
        if path.suffix == '.gz':
            text = gzip.decompress(path.read_bytes()).decode()
            json.loads(text)
        elif path.suffix in {'.py', '.sh', '.md', '.json', '.ipynb', '.txt'} or path.name == '.env.example':
            text = path.read_text()
        else:
            continue
        if TOKEN.search(text):
            failures.append(f'Credential-shaped literal; inspect without printing its value: {relative}')
        if LOCAL_PATH.search(text):
            failures.append(f'Machine-specific path: {relative}')
        if path.suffix == '.ipynb':
            notebook = json.loads(text)
            for cell in notebook['cells']:
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    failures.append(f'Notebook output was not cleared: {relative}')

    source = json.loads((ROOT / 'data/source/advbench_transformed_api_gemini-2.0-flash-001_fixed.json').read_text())
    manifest = json.loads((ROOT / 'data/reference_manifest.json').read_text())
    counts = {'generation_files': 0, 'evaluation_files': 0, 'control_task_files': 0,
              'control_result_files': 0}
    known_missing = {
        ('paper/results/gemini-2.0-flash-001/gpt-oss-120b-low/'
         'eval_level_0_64k_results_google-gemini-2.0-flash-lite-001.json', 44),
    }
    observed_missing = set()
    for entry in manifest['entries']:
        data = json.loads(gzip.decompress((ROOT / entry['archive']).read_bytes()))
        name = Path(entry['destination']).name
        if entry['destination'].startswith('paper/'):
            if name.startswith('eval_'):
                rows = data['results']
                assert len(rows) == 100, entry['destination']
                assert sorted(row['index'] for row in rows) == list(range(1, 101)), entry['destination']
                for row in rows:
                    if 'score' not in row:
                        observed_missing.add((entry['destination'], row['index']))
                    else:
                        assert row['score'] in (-1, 1, 2, 3, 4, 5), entry['destination']
                counts['evaluation_files'] += 1
            else:
                assert len(data) == 100, entry['destination']
                for i, row in enumerate(data):
                    query = source[i].get('original_query') or source[i].get('original_harmful_query')
                    assert row['sample_index'] == i, entry['destination']
                    assert row['original_task']['original_query'] == query, (entry['destination'], i)
                counts['generation_files'] += 1
        elif '/generated_tasks/' in entry['destination']:
            expected = 20 if '/benign_baseline/' in entry['destination'] else 100
            assert len(data) == expected, entry['destination']
            counts['control_task_files'] += 1
        else:
            expected = 20 if '/benign_baseline/' in entry['destination'] else 100
            rows = data['results'] if isinstance(data, dict) else data
            assert len(rows) == expected, entry['destination']
            counts['control_result_files'] += 1
    assert observed_missing == known_missing, observed_missing
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'[complete] {len(files)} publishable files; {python_count} Python and {shell_count} shell files checked')
    print('[complete] reference data alignment:', json.dumps(counts, sort_keys=True))
    print('[audit] one originally missing score retained in the supplementary 120B-low/0/64k condition; see data/README.md')
    print('[complete] no credential-shaped literals, machine-specific paths, or retained notebook outputs found')


if __name__ == '__main__':
    main()
