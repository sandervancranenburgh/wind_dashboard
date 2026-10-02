#!/usr/bin/env python3
"""Manual isolated calibration research and frozen confirmation. Never publishes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    prepare = commands.add_parser('prepare', help='Pair historical predictions and freeze a new experiment')
    prepare.add_argument('--review-dir', type=Path, required=True)
    prepare.add_argument('--reference-model-artifacts', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    confirm = commands.add_parser('confirm', help='Refresh a read-only snapshot and replay frozen models; no fitting')
    confirm.add_argument('--experiment-dir', type=Path, required=True)
    confirm.add_argument('--db', type=Path, required=True)
    confirm.add_argument('--ecmwf-archive', type=Path, required=True, help='Protected source path; not used as a learned feature')
    confirm.add_argument('--as-of', required=True, help='Timezone-aware cutoff for completed observations')
    options = parser.parse_args()
    from next_day_wind_model import d2_calibration_study as study
    study.torch.set_num_threads(2)
    if options.stage == 'prepare':
        result = study.prepare(options.review_dir.resolve(), options.reference_model_artifacts.resolve(), options.output_dir.resolve())
    else:
        result = study.confirm(options.experiment_dir.resolve(), options.db.resolve(), options.ecmwf_archive.resolve(), options.as_of)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
