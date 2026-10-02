#!/usr/bin/env python3
"""Build action-vs-normal activation vectors from collected features."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import torch

# sys.path.insert(0, str(Path(__file__).resolve().parent / 'team_code'))
from activation_steering.base import PlannerAdapter
from activation_steering.registry import get_adapter

from activation_steering.feature_collectors.base import FeatureCollector
from activation_steering.feature_collectors.default_collector import DefaultFeatureCollector
from activation_steering.feature_collectors.tfv6 import TFv6FeatureCollector


ACTION_ALIASES = {
    'brake': 'brake',
    'stop': 'brake',
    'left': 'left_change_lane',
    'left_change_lane': 'left_change_lane',
    'left-change-lane': 'left_change_lane',
    'left change lane': 'left_change_lane',
    'left_lane_change': 'left_change_lane',
    'right': 'right_change_lane',
    'right_change_lane': 'right_change_lane',
    'right-change-lane': 'right_change_lane',
    'right change lane': 'right_change_lane',
    'right_lane_change': 'right_change_lane',
}


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--adapter', default='transfuser_target_speed')
    known, _ = base_parser.parse_known_args()
    adapter = get_adapter(known.adapter)

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--collection-root',
                        default='results/steering_features')
    parser.add_argument('--logs-root', default=None,
                        help='Defaults to <collection-root>/logs.')
    parser.add_argument('--features-root', default=None,
                        help='Defaults to <collection-root>/features.')
    parser.add_argument(
        '--feature-name',
        '--feature_name',
        default=None,
        help=(
            'Optional nested feature name under each run, e.g. align_query for '
            '<features-root>/<run>/<feature-name>/<layer-name>/<frame>.pt.'
        ),
    )
    parser.add_argument(
        '--model_name',
        default=None,
        help='Optional model name to use for feature collection. If not provided, the default collector is used.',
    )
    parser.add_argument(
        '--layer-name',
        '--layer_name',
        default=None,
        help=(
            'Optional nested layer name under each feature name, e.g. layer_02 for '
            '<features-root>/<run>/<feature-name>/<layer-name>/<frame>.pt.'
        ),
    )
    parser.add_argument('--output-dir', default=None,
                        help='Defaults to <collection-root>/post_process.')
    parser.add_argument(
        '--folder-name',
        '--folder_name',
        default=None,
        help='Deprecated: subfolder under the default output root when --output-dir is not set.',
    )
    parser.add_argument(
        '--action',
        type=normalize_action,
        choices=('brake', 'left_change_lane', 'right_change_lane'),
        default='brake',
        help='Target action to use as the negative class: brake, left_change_lane, or right_change_lane.',
    )
    parser.add_argument('--model-index', type=int, default=0,
                        help='Feature subdir to use for ensembles.')
    parser.add_argument('--max-frames-per-class', type=int, default=0)
    parser.add_argument('--flatten', action='store_true',
                        help='Flatten each feature tensor before averaging.')
    parser.add_argument(
        '--manual',
        action='store_true',
        help=(
            'Use manually picked positive frames from '
            '<collection-root>/<ActionDir>/picked_frames_<feature_name>.json when '
            '--feature-name is set and that file exists, otherwise falling back to '
            '<collection-root>/<ActionDir>/picked_frames.json. Negative selection is unchanged.'
        ),
    )
    parser.add_argument(
        '--manual-negative-frames',
        '--manual_negative_frames',
        type=Path,
        default=None,
        help=(
            'Use only manually picked negative frames from this JSON file. '
            'The format is Dict[run_name, List[frame]], the same as picked_frames.json. '
            'Negative include/exclude patterns are still applied.'
        ),
    )
    parser.add_argument(
        '--force',
        '--no-cache',
        dest='force',
        action='store_true',
        help=(
            'Recompute even if a cached result already exists for this exact '
            'action/feature/layer/picked-frames combination.'
        ),
    )
    parser.add_argument('--steer-threshold', type=float, default=0.2)
    parser.add_argument('--normal-max-abs-steer', type=float, default=0.05)
    parser.add_argument(
        '--positive-include-pattern',
        action='append',
        default=[],
        help='Only use positive/target-action frames from run names or log paths containing this substring. Can be repeated.',
    )
    parser.add_argument(
        '--positive-exclude-pattern',
        action='append',
        default=[],
        help='Exclude positive/target-action frames from run names or log paths containing this substring. Can be repeated.',
    )
    parser.add_argument(
        '--negative-include-pattern',
        action='append',
        default=[],
        help='Only use negative/normal frames from run names or log paths containing this substring. Can be repeated.',
    )
    parser.add_argument(
        '--negative-exclude-pattern',
        action='append',
        default=[],
        help='Exclude negative/normal frames from run names or log paths containing this substring. Can be repeated.',
    )
    adapter.add_post_process_args(parser)
    return parser.parse_args()


def normalize_action(action: str) -> str:
    key = action.strip().lower().replace(' ', '_')
    normalized = ACTION_ALIASES.get(
        action.strip().lower(), ACTION_ALIASES.get(key))
    if normalized is None:
        options = ', '.join(('brake', 'left_change_lane', 'right_change_lane'))
        raise argparse.ArgumentTypeError(
            f"Unsupported action '{action}'. Options: {options}")
    return normalized


def matches_patterns(name: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    if include_patterns and not any(pattern in name for pattern in include_patterns):
        return False
    if any(pattern in name for pattern in exclude_patterns):
        return False
    return True


def action_dir_name(action: str) -> str:
    if action == 'brake':
        return 'Brake'
    if action == 'left_change_lane':
        return 'LaneChangeLeft'
    if action == 'right_change_lane':
        return 'LaneChangeRight'
    return action


def manual_action_dir_candidates(action: str) -> list[str]:
    candidates = [action_dir_name(action)]
    if action == 'left_change_lane':
        candidates.append('Left')
    elif action == 'right_change_lane':
        candidates.append('Right')
    return candidates


def picked_frames_filename_candidates(feature_name: str | None) -> list[str]:
    """Filenames to try, in priority order.

    When a feature name is given, prefer a feature-specific picked-frames
    file (e.g. picked_frames_decoder_layer4.json) so that different layers
    of the same action can use different hand-picked frame selections.
    Falls back to the generic picked_frames.json when no feature-specific
    file exists.
    """
    names = []
    if feature_name:
        names.append(f'picked_frames_{feature_name}.json')
    names.append('picked_frames.json')
    return names


def _find_picked_frames_path(
    collection_root: Path,
    action: str,
    include_patterns: list[str],
    filename: str,
) -> Path | None:
    candidates: list[Path] = []
    for dirname in manual_action_dir_candidates(action):
        candidates.append(collection_root / dirname / filename)

    for pattern in include_patterns:
        candidates.append(collection_root / pattern / filename)

    for path in candidates:
        if path.exists():
            return path

    matching_paths = []
    for path in sorted(collection_root.glob(f'*/{filename}')):
        match_context = f'{path.parent.name}\n{path.parent}'
        if matches_patterns(match_context, include_patterns, []):
            matching_paths.append(path)

    if len(matching_paths) == 1:
        return matching_paths[0]
    if len(matching_paths) > 1:
        shown = ', '.join(str(path) for path in matching_paths)
        raise ValueError(
            f'--manual matched multiple {filename} files: {shown}. '
            'Use a more specific --positive-include-pattern.')

    return None


def manual_positive_frame_path(
    collection_root: Path,
    action: str,
    include_patterns: list[str],
    feature_name: str | None = None,
) -> Path:
    searched: list[str] = []
    for filename in picked_frames_filename_candidates(feature_name):
        path = _find_picked_frames_path(
            collection_root, action, include_patterns, filename)
        if path is not None:
            return path
        for dirname in manual_action_dir_candidates(action):
            searched.append(str(collection_root / dirname / filename))
        for pattern in include_patterns:
            searched.append(str(collection_root / pattern / filename))

    shown = ', '.join(searched)
    raise FileNotFoundError(
        f'--manual expected picked frames at one of: {shown}')


def load_manual_frames(path: Path, option_name: str) -> dict[str, set[int]]:
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f'{option_name} picked frames must be a JSON object: {path}')

    picked: dict[str, set[int]] = {}
    for run_name, frames in raw.items():
        if not isinstance(run_name, str) or not isinstance(frames, list):
            raise ValueError(
                f'{option_name} expected Dict[str, List[int]] in {path}')
        picked[run_name] = {int(frame) for frame in frames}
    return picked


def load_manual_positive_frames(
    collection_root: Path,
    action: str,
    include_patterns: list[str],
    feature_name: str | None = None,
) -> tuple[dict[str, set[int]], Path]:
    path = manual_positive_frame_path(
        collection_root, action, include_patterns, feature_name)
    return load_manual_frames(path, '--manual'), path


def validate_manual_frame_overlap(
    positive_frames: dict[str, set[int]],
    negative_frames: dict[str, set[int]],
) -> None:
    overlaps = []
    for run_name in sorted(positive_frames.keys() & negative_frames.keys()):
        for frame in sorted(positive_frames[run_name] & negative_frames[run_name]):
            overlaps.append(f'{run_name}:{frame}')
    if overlaps:
        shown = ', '.join(overlaps[:10])
        suffix = '' if len(overlaps) <= 10 else f' ... ({len(overlaps)} total)'
        raise ValueError(
            'Frames cannot be selected as both manual positive and manual negative: '
            f'{shown}{suffix}')


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_measurements(path: Path) -> Iterable[dict]:
    for measurement in sorted(path.glob('measurements/*.json.gz')):
        frame = int(measurement.stem.split('.')[0])
        with gzip.open(measurement, 'rt', encoding='utf-8') as f:
            row = json.load(f)
        row['frame'] = frame
        yield row


def read_hipad_metas(path: Path) -> Iterable[dict]:
    for meta_path in sorted(path.glob('*.json')):
        with meta_path.open('r', encoding='utf-8') as f:
            meta = json.load(f)
        row = dict(meta)
        row['frame'] = int(meta_path.stem)
        row['_hipad_meta'] = meta
        row['_hipad_meta_path'] = str(meta_path)
        yield row


def action_logs(logs_root: Path) -> list[Path]:
    logs = sorted(logs_root.rglob('activation_actions.jsonl'))
    if logs:
        return logs
    return sorted({path.parent for path in logs_root.rglob('measurements/*.json.gz')})


def post_process_sources(logs_root: Path, collection_root: Path, adapter_name: str) -> list[Path]:
    sources = action_logs(logs_root)
    if adapter_name not in ('hipad', 'hipad_plan'):
        return sources

    existing_runs = {run_name_for(source) for source in sources}
    meta_dirs = sorted(
        path for path in collection_root.rglob('metas')
        if path.is_dir() and path.parent.parent.name == 'images'
    )
    sources.extend(path for path in meta_dirs if run_name_for(
        path) not in existing_runs)
    return sources


def run_name_for(log_source: Path) -> str:
    if log_source.name == 'metas' and log_source.parent.parent.name == 'images':
        return log_source.parent.name
    return log_source.parent.name if log_source.name == 'activation_actions.jsonl' else log_source.name


def match_context_for(log_source: Path, run_name: str) -> str:
    return f'{run_name}\n{log_source}\n{log_source.parent}'


def feature_search_root(feature_dir: Path, model_index: int) -> Path:
    model_dir = feature_dir / f'model_{model_index:02d}'
    if model_dir.exists():
        return model_dir
    return feature_dir


def feature_file_in_dir(feature_dir: Path, frame: int, model_index: int) -> Path:
    return feature_search_root(feature_dir, model_index) / f'{frame:06d}.pt'


def nested_feature_files_in_dir(
    feature_dir: Path,
    frame: int,
    model_index: int,
    feature_name: str | None,
    layer_name: str | None,
) -> list[Path]:
    search_root = feature_search_root(feature_dir, model_index)
    filename = f'{frame:06d}.pt'

    if feature_name and layer_name:
        return [search_root / feature_name / layer_name / filename]
    if feature_name:
        return sorted(path for path in (search_root / feature_name).glob(f'*/{filename}') if path.is_file())
    if layer_name:
        return sorted(path for path in search_root.glob(f'*/{layer_name}/{filename}') if path.is_file())
    return sorted(path for path in search_root.glob(f'*/*/{filename}') if path.is_file())


def feature_path_in_run_dir(
    feature_dir: Path,
    frame: int,
    model_index: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
    path = feature_file_in_dir(feature_dir, frame, model_index)
    if path.exists():
        return path

    candidates = [path for path in nested_feature_files_in_dir(
        feature_dir, frame, model_index, feature_name, layer_name) if path.exists()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        shown = ', '.join(str(path) for path in candidates[:5])
        suffix = '' if len(
            candidates) <= 5 else f', ... ({len(candidates)} total)'
        raise ValueError(
            f'Multiple feature files found for frame {frame:06d} under {feature_dir}: '
            f'{shown}{suffix}. Pass --feature-name and --layer-name to choose one.')
    return None


def sibling_feature_path_for(
    log_source: Path,
    run_name: str,
    model_index: int,
    frame: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
    if log_source.name == 'metas' and log_source.parent.parent.name == 'images':
        return feature_path_in_run_dir(
            log_source.parent.parent.parent / 'features' / run_name,
            frame,
            model_index,
            feature_name,
            layer_name,
        )

    log_dir = log_source.parent if log_source.name == 'activation_actions.jsonl' else log_source
    if log_dir.parent.name != 'logs':
        return None
    return feature_path_in_run_dir(
        log_dir.parent.parent / 'features' / run_name, frame, model_index, feature_name, layer_name)


def nested_feature_path_for(
    features_root: Path,
    run_name: str,
    model_index: int,
    frame: int,
    feature_name: str | None,
    layer_name: str | None,
) -> Path | None:
    for feature_dir in sorted(features_root.rglob(run_name)):
        if not feature_dir.is_dir():
            continue
        path = feature_path_in_run_dir(
            feature_dir, frame, model_index, feature_name, layer_name)
        if path is not None:
            return path
    return None


def feature_path_for(
    row: dict,
    log_source: Path,
    logs_root: Path,
    features_root: Path,
    args: argparse.Namespace,
    adapter: PlannerAdapter,
) -> Path:
    run_name = run_name_for(log_source)
    frame = int(row['frame'])
    action: str = args.action

    # 1. Direct path from log record if explicitly given and no feature_name override is requested
    if row.get('feature_path'):
        p = Path(row['feature_path'])
        if p.exists():
            return p

    # 2. Resolve via FeatureCollector path standard: <features_root>/<run_name>/<feature_name>/...
    features_root = features_root / action_dir_name(action) / 'features'
    run_feature_dir = features_root / run_name

    feature_collector: FeatureCollector
    if args.model_name == 'tfv6':
        feature_collector = TFv6FeatureCollector(output_root=run_feature_dir,)
    else:
        feature_collector = DefaultFeatureCollector(output_root=run_feature_dir)

    candidate = feature_collector.resolve_feature_path(
        feature_dir=run_feature_dir,
        frame_idx=frame,
        feature_name=args.feature_name,
        model_idx=args.model_index,
    )

    if candidate.exists():
        return candidate

    # 3. Fallback to adapter
    print("Falling back to adapter for feature path resolution.")
    return adapter.feature_path_for(row, log_source, features_root, args.model_index)


def _extract_model_index_from_path(path: Path) -> int:
    """
    Extracts the model index from the feature path based on the expected directory structure.
    If the model index cannot be determined, it defaults to -1.
    Example:
        /mnt/bapve/thome/wangyuhao/fail2drive/steering/tfv6/Normal/features/Normal_0035_route0_08_28_23_46_56/model_01/fused_features/000335.pt
        -> 1
    """
    parts = path.parts
    for _, part in enumerate(parts):
        if part.startswith('model_'):
            try:
                return int(part.split('_')[1])
            except (IndexError, ValueError):
                return -1
    return -1


def default_child_or_root(root: Path, child_name: str) -> Path:
    child = root / child_name
    return child if child.exists() else root


CACHE_FILENAME = '.process_cache.json'
REQUIRED_OUTPUT_FILENAMES = (
    'positive_mean.pt', 'negative_mean.pt', 'steering_vector.pt', 'summary.json',
)


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def frames_signature(frames: dict[str, set[int]]) -> str:
    """Order-independent content hash of a picked-frames mapping.

    Used instead of (or in addition to) a file hash so that the cache
    stays valid even if the picked_frames JSON is rewritten with the same
    content in a different key/list order.
    """
    import hashlib
    h = hashlib.sha256()
    for run_name in sorted(frames.keys()):
        h.update(run_name.encode('utf-8'))
        h.update(b'\x00')
        for frame in sorted(frames[run_name]):
            h.update(str(frame).encode('utf-8'))
            h.update(b',')
        h.update(b'\x01')
    return h.hexdigest()


def output_dirs_for(output_dir: Path, model_count: int, feature_name: str | None) -> list[Path]:
    dirs = []
    for model_idx in range(model_count):
        current = output_dir / f'model_{model_idx:02d}' if model_count > 1 else output_dir
        if feature_name:
            current = current / feature_name
        dirs.append(current)
    return dirs


def build_run_signature(
    args: argparse.Namespace,
    manual_path: Path | None,
    manual_frames: dict[str, set[int]],
    manual_negative_path: Path | None,
    manual_negative_frames: dict[str, set[int]] | None,
) -> dict[str, Any]:
    return {
        'version': 1,
        'adapter': args.adapter,
        'model_name': args.model_name,
        'action': args.action,
        'feature_name': args.feature_name,
        'layer_name': args.layer_name,
        'model_index': args.model_index,
        'max_frames_per_class': args.max_frames_per_class,
        'flatten': args.flatten,
        'manual': args.manual,
        'steer_threshold': args.steer_threshold,
        'normal_max_abs_steer': args.normal_max_abs_steer,
        'positive_include_pattern': sorted(args.positive_include_pattern),
        'positive_exclude_pattern': sorted(args.positive_exclude_pattern),
        'negative_include_pattern': sorted(args.negative_include_pattern),
        'negative_exclude_pattern': sorted(args.negative_exclude_pattern),
        'collection_root': str(args.collection_root),
        'logs_root': str(args.logs_root) if args.logs_root else None,
        'features_root': str(args.features_root) if args.features_root else None,
        'manual_positive_frames_path': str(manual_path) if manual_path else None,
        'manual_positive_frames_content_hash': (
            frames_signature(manual_frames) if manual_path else None
        ),
        'manual_positive_frames_file_hash': (
            file_sha256(manual_path) if manual_path else None
        ),
        'manual_negative_frames_path': str(manual_negative_path) if manual_negative_path else None,
        'manual_negative_frames_content_hash': (
            frames_signature(manual_negative_frames)
            if manual_negative_frames is not None else None
        ),
        'manual_negative_frames_file_hash': (
            file_sha256(manual_negative_path) if manual_negative_path else None
        ),
    }


def cache_is_fresh(current_output_dir: Path, signature: dict[str, Any]) -> bool:
    cache_path = current_output_dir / CACHE_FILENAME
    if not cache_path.exists():
        return False
    for filename in REQUIRED_OUTPUT_FILENAMES:
        if not (current_output_dir / filename).exists():
            return False
    try:
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return False
    return cached == signature


def write_cache(current_output_dir: Path, signature: dict[str, Any]) -> None:
    cache_path = current_output_dir / CACHE_FILENAME
    cache_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')


def add_feature(accumulator: dict, label: str, feature: torch.Tensor, max_count: int) -> bool:
    if max_count > 0 and accumulator[f'{label}_count'] >= max_count:
        return False

    feature = feature.detach().cpu().float()
    if accumulator['flatten']:
        feature = feature.reshape(-1)

    key = f'{label}_sum'
    if accumulator[key] is None:
        accumulator[key] = torch.zeros_like(feature)
    accumulator[key] += feature
    accumulator[f'{label}_count'] += 1
    return True


def basic_frame_allowed(row: dict, args: argparse.Namespace) -> bool:
    speed = float(row.get('speed', 0.0))
    stop_for_stop_sign = bool(
        row.get('stop_for_stop_sign', row.get('stop_sign_hazard', False)))
    if args.exclude_stop_sign and stop_for_stop_sign:
        return False
    return speed >= args.min_speed


def classify_normal(row: dict, match_context: str, args: argparse.Namespace) -> str | None:
    if not matches_patterns(match_context, args.negative_include_pattern, args.negative_exclude_pattern):
        return None
    if not matches_patterns(match_context, args.normal_include_pattern, args.normal_exclude_pattern):
        return None
    if not basic_frame_allowed(row, args):
        return None

    steer = float(row.get('steer', 0.0))
    throttle = float(row.get('throttle', 0.0))
    brake = float(row.get('brake', row.get('control_brake', 0.0)))
    if abs(steer) <= args.normal_max_abs_steer and brake < args.brake_threshold and throttle >= args.normal_throttle_threshold:
        return 'negative'
    return None


def classify_target_action(row: dict, run_name: str, match_context: str, args: argparse.Namespace, adapter) -> str | None:
    if not matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
        return None
    if not basic_frame_allowed(row, args):
        return None

    if args.action == 'brake':
        if not matches_patterns(match_context, args.brake_include_pattern, args.brake_exclude_pattern):
            return None
        if adapter.classify_frame(row, run_name, args) == 'brake':
            return 'positive'
        return None

    steer = float(row.get('steer', 0.0))
    brake = float(row.get('brake', row.get('control_brake', 0.0)))
    if brake >= args.brake_threshold:
        return None
    if args.action == 'right_change_lane' and steer > args.steer_threshold:
        return 'positive'
    if args.action == 'left_change_lane' and steer < -args.steer_threshold:
        return 'positive'
    return None


def classify_frame(row: dict, run_name: str, match_context: str, args: argparse.Namespace, adapter) -> str | None:
    frame = int(row['frame'])
    if args.manual:
        manual_frames = getattr(args, '_manual_positive_frames', {})
        if frame in manual_frames.get(run_name, set()):
            if matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
                return adapter.filter_post_process_label(row, 'positive', args.action, run_name, args)
            return None

    manual_negative_frames = getattr(args, '_manual_negative_frames', None)
    if manual_negative_frames is not None:
        if frame in manual_negative_frames.get(run_name, set()):
            if matches_patterns(match_context, args.negative_include_pattern, args.negative_exclude_pattern):
                if matches_patterns(match_context, args.normal_include_pattern, args.normal_exclude_pattern):
                    return adapter.filter_post_process_label(row, 'negative', args.action, run_name, args)
            return None

    custom_label = adapter.classify_post_process_label(
        row, args.action, run_name, args)
    if custom_label == 'skip':
        return None
    if custom_label == 'positive':
        if args.manual:
            return None
        if matches_patterns(match_context, args.positive_include_pattern, args.positive_exclude_pattern):
            return adapter.filter_post_process_label(row, 'positive', args.action, run_name, args)
        return None
    if custom_label == 'negative':
        if manual_negative_frames is not None:
            return None
        if matches_patterns(match_context, args.negative_include_pattern, args.negative_exclude_pattern):
            if matches_patterns(match_context, args.normal_include_pattern, args.normal_exclude_pattern):
                return adapter.filter_post_process_label(row, 'negative', args.action, run_name, args)
        return None

    if not args.manual:
        positive = classify_target_action(
            row, run_name, match_context, args, adapter)
        if positive is not None:
            return adapter.filter_post_process_label(row, positive, args.action, run_name, args)
    if manual_negative_frames is not None:
        return None
    normal = classify_normal(row, match_context, args)
    if normal is not None:
        return adapter.filter_post_process_label(row, normal, args.action, run_name, args)
    return None


def main() -> int:
    args = parse_args()
    adapter = get_adapter(args.adapter)
    collection_root = Path(args.collection_root)
    logs_root = Path(args.logs_root) if args.logs_root else default_child_or_root(
        collection_root, 'logs')
    features_root = Path(args.features_root) if args.features_root else default_child_or_root(
        collection_root, 'features')
    output_dir = Path(args.output_dir) if args.output_dir else collection_root / \
        'post_process' / (args.folder_name or args.action)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_count: int = 3

    manual_path = None
    manual_frames: dict[str, set[int]] = {}
    if args.manual:
        manual_frames, manual_path = load_manual_positive_frames(
            collection_root, args.action, args.positive_include_pattern, args.feature_name)
        setattr(args, '_manual_positive_frames', manual_frames)

    manual_negative_path = args.manual_negative_frames
    manual_negative_frames = None
    if manual_negative_path is not None:
        if not manual_negative_path.exists():
            raise FileNotFoundError(
                f'--manual-negative-frames file does not exist: {manual_negative_path}')
        manual_negative_frames = load_manual_frames(
            manual_negative_path, '--manual-negative-frames')
        validate_manual_frame_overlap(manual_frames, manual_negative_frames)
        setattr(args, '_manual_negative_frames', manual_negative_frames)

    signature = build_run_signature(
        args, manual_path, manual_frames, manual_negative_path, manual_negative_frames)
    expected_output_dirs = output_dirs_for(output_dir, model_count, args.feature_name)

    if not args.force and expected_output_dirs and all(
        cache_is_fresh(d, signature) for d in expected_output_dirs
    ):
        print(
            f'Skipping (cache hit): action={args.action} feature_name={args.feature_name} '
            f'-- picked frames and config unchanged since last run. '
            f'Pass --force to recompute.'
        )
        for d in expected_output_dirs:
            print(f'  up to date: {d}')
        return 0

    accumulators = [{
        'positive_sum': None,
        'negative_sum': None,
        'positive_count': 0,
        'negative_count': 0,
        'flatten': args.flatten,
    } for _ in range(model_count)]
    manifest_path = output_dir / 'selected_frames.jsonl'
    missing_features = 0
    total_rows = 0

    with manifest_path.open('w', encoding='utf-8') as manifest:
        for log_source in post_process_sources(logs_root, collection_root, args.adapter):
            run_name = run_name_for(log_source)
            match_context = match_context_for(log_source, run_name)
            if log_source.name == 'activation_actions.jsonl':
                rows_iter = read_jsonl(log_source)
            elif log_source.name == 'metas':
                rows_iter = read_hipad_metas(log_source)
            else:
                rows_iter = read_measurements(log_source)
            rows = adapter.augment_rows(
                list(rows_iter), log_source, collection_root)
            rows = adapter.annotate_rows(rows, args)
            for row in rows:
                total_rows += 1
                label = classify_frame(
                    row, run_name, match_context, args, adapter)
                if label is None:
                    continue

                path = feature_path_for(
                    row, log_source, logs_root, features_root, args, adapter)

                model_idx = _extract_model_index_from_path(path)
                if model_idx >= model_count:
                    raise ValueError(
                        f"Extracted model index {model_idx} from path {path} is out of bounds (0-{model_count-1})")

                if not path.exists():
                    missing_features += 1
                    continue

                feature = adapter.load_feature(path)
                indices_to_add = [model_idx]  # if label == 'positive' else range(model_count)

                # record: dict[str, Any] = {}
                for idx in indices_to_add:
                    if add_feature(accumulators[idx], label, feature, args.max_frames_per_class):
                        record = {
                            'label': label,
                            'action': args.action,
                            'run_name': run_name,
                            'frame': int(row['frame']),
                            'model_idx': idx,
                            'feature_path': str(path),
                            'speed': float(row.get('speed', 0.0)),
                            'steer': float(row.get('steer', 0.0)),
                        }
                        record.update(adapter.manifest_extra(row))
                        manifest.write(json.dumps(record) + '\n')

    for model_idx in range(model_count):
        if model_count > 1:
            current_output_dir = output_dir / f'model_{model_idx:02d}'
        else:
            current_output_dir = output_dir

        if args.feature_name:
            current_output_dir = current_output_dir / args.feature_name

        current_output_dir.mkdir(parents=True, exist_ok=True)

        if accumulators[model_idx]['positive_count'] == 0 or accumulators[model_idx]['negative_count'] == 0:
            raise RuntimeError(
                f"Need both classes, got positive={accumulators[model_idx]['positive_count']} "
                f"negative={accumulators[model_idx]['negative_count']} missing_features={missing_features}")

        positive_mean = accumulators[model_idx]['positive_sum'] / accumulators[model_idx]['positive_count']
        negative_mean = accumulators[model_idx]['negative_sum'] / accumulators[model_idx]['negative_count']
        steering_vector = positive_mean - negative_mean

        torch.save(positive_mean, current_output_dir / 'positive_mean.pt')
        torch.save(negative_mean, current_output_dir / 'negative_mean.pt')
        torch.save(steering_vector, current_output_dir / 'steering_vector.pt')

        summary = {
            'adapter': args.adapter,
            'action': args.action,
            'positive_label': args.action,
            'negative_label': 'normal',
            'positive_count': accumulators[model_idx]['positive_count'],
            'negative_count': accumulators[model_idx]['negative_count'],
            'total_rows': total_rows,
            'missing_features': missing_features,
            'flatten': args.flatten,
            'vector_formula': 'positive_mean - negative_mean',
            'manual_positive_frames_path': None if manual_path is None else str(manual_path),
            'manual_positive_frame_count': (
                0 if manual_path is None
                else sum(len(frames) for frames in getattr(args, '_manual_positive_frames', {}).values())
            ),
            'manual_negative_frames_path': (
                None if manual_negative_path is None else str(manual_negative_path)
            ),
            'manual_negative_frame_count': (
                0 if manual_negative_frames is None
                else sum(len(frames) for frames in manual_negative_frames.values())
            ),
            'output_files': {
                'positive_mean': str(current_output_dir / 'positive_mean.pt'),
                'negative_mean': str(current_output_dir / 'negative_mean.pt'),
                'steering_vector': str(current_output_dir / 'steering_vector.pt'),
                'selected_frames': str(manifest_path),
            },
            'args': {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.startswith('_')
            },
        }
        (current_output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        write_cache(current_output_dir, signature)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
