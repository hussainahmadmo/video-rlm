#!/usr/bin/env python3
"""Create matched balanced and noisy-neighbor traces for fairness/efficiency."""
import argparse
import json
import random
from pathlib import Path


def row(request_id, tenant, frames, arrival):
    return {
        'request_id': request_id,
        'qid': request_id,
        'tenant': tenant,
        'modality': 'video',
        'video': 'replaced-by-runner',
        'frame_count': frames,
        'arrival_s': arrival,
        'max_tokens': 32,
        'priority': 0,
        'prompt_override': 'Briefly describe the video.',
        'class': 'background',
    }


def balanced():
    rng = random.Random(11)
    rows = []
    for tenant in ('a', 'b', 'c'):
        sizes = [1] * 5 + [16] * 10 + [128] * 5
        rng.shuffle(sizes)
        for index, frames in enumerate(sizes):
            rows.append(row(f'{tenant}-{index}', tenant, frames,
                            rng.uniform(0, 15)))
    return sorted(rows, key=lambda item: (item['arrival_s'], item['request_id']))


def noisy_neighbor():
    rows = [row(f'a-{index}', 'a', 128, index * .075)
            for index in range(40)]
    rows += [row(f'b-{index}', 'b', 1, .5 + index * .6)
             for index in range(10)]
    rows += [row(f'c-{index}', 'c', 16, .65 + index * .6)
             for index in range(10)]
    return sorted(rows, key=lambda item: (item['arrival_s'], item['request_id']))


def write(path, rows):
    path.write_text(''.join(json.dumps(item) + '\n' for item in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / 'balanced.jsonl', balanced())
    write(args.output_dir / 'noisy_neighbor.jsonl', noisy_neighbor())


if __name__ == '__main__':
    main()
