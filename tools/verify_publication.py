"""Verify the public export hashes, common secret patterns and local path redaction."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import zipfile

PRIVATE_PATH = re.compile(r'(?i)\b[A-Z]:(?:\\+|/)(?:Users|AS|Claudecode_workspace|codex_workspace)(?:\\+|/)')
SECRET_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{32,}|xox[baprs]-[A-Za-z0-9-]{20,})'),
    'credential_url': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'cds_uid_key': re.compile(r'(?m)^\s*key:\s*\d{3,}:[A-Za-z0-9-]{20,}'),
}

def check_text(name, data):
    if b'\0' in data[:8192]:
        return []
    try:
        value = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return []
    problems = []
    for kind, pattern in {'private_machine_path': PRIVATE_PATH, **SECRET_PATTERNS}.items():
        for match in pattern.finditer(value):
            problems.append({'path': name, 'line': value.count('\n', 0, match.start()) + 1,
                             'kind': kind})
    return problems

def verify(root):
    manifest_path = root / 'publication/publication_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors, warnings = [], []
    count = 0
    for row in manifest['files']:
        if row['published_sha256'] is None:
            continue
        name = row['path']
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Publication entry escapes repository')
        if not path.is_file():
            errors.append({'path': name, 'kind': 'missing_file'})
            continue
        data = path.read_bytes()
        count += 1
        if hashlib.sha256(data).hexdigest() != row['published_sha256']:
            errors.append({'path': name, 'kind': 'hash_mismatch'})
        if len(data) > 100 * 1024**2:
            errors.append({'path': name, 'kind': 'over_100_MiB'})
        elif len(data) > 50 * 1024**2:
            warnings.append({'path': name, 'kind': 'over_50_MiB', 'bytes': len(data)})
        errors.extend(check_text(name, data))
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        errors.extend(check_text(name + '!' + member.filename, archive.read(member)))
    errors.extend(check_text(manifest_path.relative_to(root).as_posix(), manifest_path.read_bytes()))
    result = {'status': 'pass' if not errors else 'fail', 'verified_files': count,
              'source_commit': manifest['source_commit'],
              'errors': errors, 'warnings': warnings,
              'scope': 'Published files and embedded ZIP members; common patterns, not a guarantee against every secret format.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = verify(args.root.resolve())
    raise SystemExit(0 if result['status'] == 'pass' else 1)

if __name__ == '__main__':
    main()
