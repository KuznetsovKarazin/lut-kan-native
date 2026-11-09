"""Audit the three original V2 ZIPs and reproduce the frozen hardware summary.

No fitting, parameter selection or hardware acquisition is performed.
Example: python analyze_v2_captures.py --archives raw_archives --out v2_audit
"""
import argparse
import hashlib
import json
import re
import statistics
import struct
import zipfile
from pathlib import Path

from mcu.capture_references import METHODS, parse_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARATION = HERE / 'results/acquisition_preparation_20261109'
INPUTS = {'mega2560': 'mega_v2_20261109.zip',
          'esp32c3': 'c3_v2_20261109.zip',
          'esp32s3': 's3_v2_20261109.zip'}
EXPECTED_ARCHIVES = {
    'mega2560': '24dd8490806d83e43fe3f5c9320f5b0aeefd4e2e9c4813685c76303290db5c32',
    'esp32c3': '571a90e2d3f68145b17b8070c1a359ec50e3f25101f0c1d399cbfd3be08825ec',
    'esp32s3': '491bb9c975c1302f22329b1efe316e95d64d5ceff1f5f1d506d62bbbf326ee02'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    return json.loads(data.decode('utf-8-sig'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def verify_s3_float(elf):
    """Check captured ELF opcode bytes against the pre-acquisition disassembly.

    ELF32 section/symbol parsing uses only the Python standard library. This is
    byte verification of identified instructions, not a new disassembler.
    """
    header = struct.unpack_from('<16sHHIIIIIHHHHHH', elf)
    require(header[0][:6] == b'\x7fELF\x01\x01', 'Expected little-endian ELF32')
    sections = [struct.unpack_from('<IIIIIIIIII', elf, header[6] + i * header[11])
                for i in range(header[12])]
    functions = {}
    for section in sections:
        if section[1] != 2:
            continue
        strings_section = sections[section[6]]
        strings = elf[strings_section[4]:strings_section[4] + strings_section[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, value, size, info, _, index = struct.unpack_from('<IIIBBH', elf, offset)
            name = strings[name:].split(b'\0', 1)[0].decode()
            if 'predict_' in name and info & 15 == 2:
                target = sections[index]
                position = target[4] + value - target[3]
                functions[name] = (value, elf[position:position + size])
    report = {}
    text = (PREPARATION / 's3_predictors_disassembly.txt').read_text()
    for block in re.split(r'\n\n+', text):
        lines = block.splitlines()
        if not lines:
            continue
        match = re.match(r'([0-9a-f]+) <(predict_[1-7])\(float\)>:', lines[0])
        if not match:
            continue
        original_base, name = int(match[1], 16), match[2]
        symbol = next(s for s in functions if name in s)
        address, code = functions[symbol]
        instructions = []
        for line in lines[1:]:
            match = re.match(r'([0-9a-f]+):\s+([0-9a-f]+)\s+((?:add|sub|mul|mov|float)\.s)\b', line)
            if not match:
                continue
            offset = int(match[1], 16) - original_base
            expected = bytes.fromhex(match[2])[::-1]
            actual = code[offset:offset + len(expected)]
            require(actual == expected, 'FPU opcode mismatch: ' + name)
            instructions.append(dict(offset=offset, instruction=match[3], bytes=actual.hex()))
        require(bool(instructions), 'No verified float instructions: ' + name)
        report[name] = dict(symbol=symbol, address=hex(address), size=len(code),
                            verified_float_instructions=instructions)
    require(len(report) == 7, 'Missing S3 predictors')
    return report


def analyze(archives, out):
    require(not out.exists(), 'Choose a new output directory')
    out.mkdir(parents=True)
    expected_sources = decode((PREPARATION / 'build_audit.json').read_bytes())['source_hashes']
    all_sessions, boards, archive_manifest, fpu = set(), {}, {}, None
    for board, filename in INPUTS.items():
        path = archives / filename
        archive_manifest[filename] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
        require(archive_manifest[filename]['sha256'] == EXPECTED_ARCHIVES[board], 'Archive identity changed')
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, 'ZIP integrity failure')
            require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP entries')
            files = {n: archive.read(n) for n in archive.namelist()}
        batch = decode(files['batch.json'])
        require((batch['board'], batch['status'], batch['stage'], batch['requested_runs']) ==
                (board, 'validated', 'complete', 2), 'Invalid batch identity/status')
        sources = {p.replace('\\', '/'): r for p, r in batch['source_before'].items()}
        require(batch['source_before'] == batch['source_after'], 'Source changed during acquisition')
        require(sources == expected_sources, 'Sources differ from frozen preparation')
        for name, record in sources.items():
            require(sha((ROOT / name).read_bytes()) == record['raw_sha256'], 'Repository source mismatch')
        for record in batch['built_artifacts'].values():
            data = files[record['archive_path']]
            require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'Firmware hash mismatch')
        require('SUCCESS' in files['upload.log'].decode(errors='replace'), 'No successful upload log')
        if board == 'esp32s3':
            fpu = verify_s3_float(files['firmware/firmware.elf'])
        runs, sessions = [], []
        require(len(batch['captures']) == 2, 'Wrong batch capture count')
        for i, item in enumerate(batch['captures'], 1):
            stem = f'run{i:02d}'
            result = decode(files[stem + '.json'])
            side = decode(files[stem + '.session.json'])
            parsed = parse_lines(files[stem + '.log'].decode('utf-8-sig').splitlines())
            require(parsed == result, 'Raw log does not reproduce JSON')
            token = result['metadata']['SESSION']
            require(result['metadata']['BOARD'] == board and token not in all_sessions, 'Board/session collision')
            require(side['session_id'] == token and item['session_id'] == token, 'Session identity mismatch')
            require(side['status'] == 'validated' and side['usb_phase'] == 'complete', 'Incomplete sidecar')
            require(side['board_requested'] == board, 'Wrong requested board')
            require(side['capture_script_sha256'] == sources['experiments/sensor_recovery/mcu/capture_references.py']['raw_sha256'], 'Capture script mismatch')
            for key, suffix in [('raw_sha256', '.log'), ('result_sha256', '.json')]:
                require(sha(files[stem + suffix]) == side[key] == item[key], 'Capture hash mismatch')
            require(item['result'] == stem + '.json' and side['raw_bytes'] == len(files[stem + '.log']), 'Capture metadata mismatch')
            all_sessions.add(token)
            runs.append(result)
            sessions.append(dict(session_id=token, start_utc=side['start_utc'], end_utc=side['end_utc'],
                                 pre_ready_rejections=len(side['pre_ready_rejections'])))
        methods = {}
        for method in ('overhead', *METHODS):
            times = [run['inference_median_us'][method] for run in runs]
            med = statistics.median(times)
            batches = [[r['total_us'] / r['calls'] for r in run['timings'] if r['method'] == method] for run in runs]
            record = dict(median_of_session_medians_us=med, session_medians_us=times,
                          session_range_us=max(times) - min(times),
                          relative_session_range_pct=100 * (max(times) - min(times)) / med,
                          batch_us=batches, batch_min_us=min(sum(batches, [])), batch_max_us=max(sum(batches, [])))
            if method != 'overhead':
                record.update(payload_bytes=runs[0]['payload_bytes'][method],
                              metrics=[run['accuracy'][method] for run in runs],
                              mae_C=statistics.median(run['accuracy'][method]['mae_C'] for run in runs),
                              max_error_C=max(run['accuracy'][method]['max_C'] for run in runs))
            methods[method] = record
        for method in METHODS:
            current = methods[method]
            current['dominated_by'] = [name for name in METHODS if name != method and
                methods[name]['median_of_session_medians_us'] <= current['median_of_session_medians_us'] and
                methods[name]['mae_C'] <= current['mae_C'] and
                (methods[name]['median_of_session_medians_us'] < current['median_of_session_medians_us'] or
                 methods[name]['mae_C'] < current['mae_C'])]
        boards[board] = dict(cpu_hz=int(runs[0]['metadata']['CPU_HZ']), sessions=sessions,
                             methods=methods, frontier=[m for m in METHODS if not methods[m]['dominated_by']],
                             source_files_verified=len(sources), firmware_images_verified=len(batch['built_artifacts']))
        # Publish textual evidence; full original ZIPs preserve firmware and upload logs.
        target = out / 'captures' / board
        target.mkdir(parents=True)
        for name, data in files.items():
            if name in ('batch.json', 'packages.txt') or re.fullmatch(r'run0[12]\.(?:log|json|session\.json|console\.txt)', name):
                (target / name).write_bytes(data)
    result = dict(status='PASS', acquisition_date='2026-11-09',
                  acquisition_source_commit='755ffe07d04fe053f8e626a54b1ec8637dbf2ba2',
                  suite='matched_references_v2', fixture_sha256=decode((HERE / 'mcu/revision_manifest.json').read_bytes())['fixture_sha256'],
                  seed=2026110801, scenario='ntc_frontend', unit=0, repeat=0, budget=75,
                  total_sessions=6, total_timing_records=480, total_accuracy_records=42,
                  batches_per_method_per_session=10, calls_per_batch=2048,
                  aggregation='Median of ten batch times per session, then median of two session medians; gross times; no overhead subtraction.',
                  scope='Measured execution on three physical MCU boards; fixed synthetic accuracy fixture; no new fitting or hyperparameter selection.',
                  limits='One board per architecture; no sensor measurements, energy, WCET, peak stack or between-board uncertainty. Clock/ISA/memory/compiler also differ. Exact MAE differences between polynomial forms are numerical rounding, not an accuracy benefit.',
                  max_host_gap_C=max(a['max_host_gap_C'] for b in boards.values() for m in METHODS for a in b['methods'][m]['metrics']),
                  boards=boards, archives=archive_manifest)
    dump(out / 'summary.json', result)
    dump(out / 's3_captured_elf_fpu.json', fpu)
    manifest = {p.relative_to(out).as_posix(): sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out / 'SHA256.json', manifest)
    print('PASS: 6 sessions; 480 timing records; 42 accuracy records; sources/images/sidecars verified.')
    for board, data in boards.items():
        print(board, 'frontier:', ', '.join(data['frontier']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.archives, args.out)
