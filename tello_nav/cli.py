import argparse
import hashlib
import json
from pathlib import Path
import sys

from tello_nav.controller import ControllerConfig
from tello_nav.controller.sectors import SectorConfig
from tello_nav.replay.logging import write_results
from tello_nav.replay.providers import MockDepthProvider, NpyDepthProvider, synthetic_frames
from tello_nav.replay.runner import ReplayConfig, ReplayRunner
from tello_nav.replay.video import VideoReader


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless by default; replay cannot transmit to a real drone")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--synthetic", choices=MockDepthProvider.SCENARIOS)
    source.add_argument("--manifest", type=Path, help="JSON list of recorded scenarios, paths relative to manifest")
    parser.add_argument("--timestamps", type=Path)
    parser.add_argument("--depth-dir", type=Path)
    parser.add_argument("--predictions", type=Path, help="versioned external prediction manifest")
    parser.add_argument("--calibration-id", help="expected prediction calibration identifier")
    parser.add_argument("--camera-id", help="expected prediction camera identifier")
    parser.add_argument("--allow-synthetic-predictions", action="store_true",
                        help="explicitly permit synthetic exchange fixtures, labeled unsuitable for metric evaluation")
    parser.add_argument("--mock-depth", choices=MockDepthProvider.SCENARIOS)
    parser.add_argument("--depth-model", choices=("da3", "metric3d"), help="use the team's existing RGB/focal wrapper")
    parser.add_argument("--calibration", type=Path, help="size/K/dist JSON for this camera and image geometry")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true",
                        help="authorize model constructors to fetch external code/weights")
    parser.add_argument("--mock-drone", action="store_true", help="execute decisions through the mock adapter only")
    parser.add_argument("--model-name", default="precomputed")
    parser.add_argument("--scenario-id", default="scenario")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--mode", choices=("offline", "virtual", "realtime"), default="offline")
    parser.add_argument("--inference-delay", type=float, default=0.0)
    parser.add_argument("--communication-delay", type=float, default=0.0)
    parser.add_argument("--late-policy", choices=("process", "drop"), default="process")
    parser.add_argument("--max-lateness", type=float, default=0.5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--step", action="store_true", help="press Enter before requesting each next decision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.synthetic and (args.timestamps or args.depth_dir or args.mock_depth or args.depth_model or args.calibration or args.predictions):
            raise ValueError("synthetic source supplies its own frames and depth")
        if args.manifest and (args.timestamps or args.depth_dir or args.mock_depth or args.depth_model or args.calibration or args.predictions or args.calibration_id or args.camera_id):
            raise ValueError("manifest entries supply video, timestamps and depth source")
        if args.step and args.mode == "realtime":
            raise ValueError("step mode is offline/virtual only")
        data = json.loads(args.controller_config.read_text()) if args.controller_config else {}
        if "sectors" in data:
            data["sectors"] = SectorConfig(**data["sectors"])
        controller = ControllerConfig(**data)
        config = ReplayConfig(args.mode, args.inference_delay, args.communication_delay,
                              args.late_policy, args.max_lateness)
        specs = []
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
            if not isinstance(manifest, list) or not manifest:
                raise ValueError("manifest must be a nonempty list")
            ids = set()
            for spec in manifest:
                spec = dict(spec)
                if not isinstance(spec.get("id"), str) or not spec["id"] or spec["id"] in ids:
                    raise ValueError("manifest scenario IDs must be nonempty and unique")
                ids.add(spec["id"])
                for key in ("video", "timestamps", "depth_dir", "calibration", "predictions"):
                    if key in spec:
                        spec[key] = args.manifest.parent / spec[key]
                specs.append(spec)
        else:
            specs = [{"id": args.scenario_id, "synthetic": args.synthetic, "video": args.video,
                      "timestamps": args.timestamps, "depth_dir": args.depth_dir,
                      "mock_depth": args.mock_depth, "model_name": args.model_name,
                      "depth_model": args.depth_model, "calibration": args.calibration,
                      "predictions": args.predictions, "calibration_id": args.calibration_id,
                      "camera_id": args.camera_id}]
        records, scenarios = [], []
        model_providers = {}
        for spec in specs:
            inputs = []
            if spec.get("synthetic"):
                if any(spec.get(key) for key in ("video", "timestamps", "depth_dir", "mock_depth", "depth_model", "calibration", "predictions", "calibration_id", "camera_id")):
                    raise ValueError("synthetic scenario cannot also specify a recorded source or model")
                frames = synthetic_frames(args.frames)
                provider = MockDepthProvider(spec["synthetic"])
            else:
                if not spec.get("video"):
                    raise ValueError("recorded scenario needs video")
                frames = VideoReader(spec["video"], spec.get("timestamps"))
                for key in ("video", "timestamps"):
                    if spec.get(key):
                        inputs.append(fingerprint(Path(spec[key])))
                if sum(bool(spec.get(key)) for key in ("depth_dir", "mock_depth", "depth_model", "predictions")) != 1:
                    raise ValueError("choose exactly one of depth_dir, mock_depth, depth_model or predictions for video")
                if spec.get("calibration") and not (spec.get("depth_model") or spec.get("predictions")):
                    raise ValueError("calibration is only used with depth_model or predictions")
                if (spec.get("calibration_id") or spec.get("camera_id")) and not spec.get("predictions"):
                    raise ValueError("camera/calibration identifiers are only used with predictions")
                if spec.get("depth_dir"):
                    provider = NpyDepthProvider(spec["depth_dir"], spec.get("model_name", "precomputed"))
                    inputs.extend(fingerprint(p) for p in sorted(Path(spec["depth_dir"]).glob("*.npy")))
                elif spec.get("predictions"):
                    if not all(spec.get(key) for key in ("calibration", "calibration_id", "camera_id")):
                        raise ValueError("predictions require explicit calibration, calibration_id and camera_id")
                    from tello_nav.replay.predictions import ManifestDepthProvider
                    provider = ManifestDepthProvider(
                        spec["predictions"], scenario_id=spec.get("scenario_id", spec["id"]), video_path=spec["video"],
                        calibration_path=spec["calibration"], calibration_id=spec["calibration_id"],
                        camera_id=spec["camera_id"], allow_synthetic=args.allow_synthetic_predictions)
                    inputs.extend(fingerprint(Path(spec[key])) for key in ("predictions", "calibration"))
                elif spec.get("depth_model"):
                    if not spec.get("calibration"):
                        raise ValueError("depth_model requires an explicit calibration for this camera")
                    from tello_nav.replay.model_provider import create_model_provider
                    inputs.append(fingerprint(Path(spec["calibration"])))
                    provider_key = (spec["depth_model"], str(Path(spec["calibration"]).resolve()), args.device)
                    if provider_key not in model_providers:
                        model_providers[provider_key] = create_model_provider(
                            spec["depth_model"], spec["calibration"], device=args.device,
                            allow_model_download=args.allow_model_download)
                    provider = model_providers[provider_key]
                else:
                    provider = MockDepthProvider(spec["mock_depth"])
            runner = ReplayRunner(provider, controller, config, mock_drone=args.mock_drone)
            iterator = runner.iter_records(frames, spec["id"], args.visualize)
            try:
                while True:
                    if args.step:
                        input("Enter for next frame (Ctrl-C to stop): ")
                    try:
                        record = next(iterator)
                    except StopIteration:
                        break
                    records.append(record)
                    if args.step:
                        print(json.dumps(record))
            finally:
                iterator.close()
            scenarios.append({"id": spec["id"], "inputs": inputs,
                              "synthetic_scenario": spec.get("synthetic") or spec.get("mock_depth"),
                              "synthetic_frames": args.frames if spec.get("synthetic") else None,
                              **runner.metadata()})
        import cv2
        import numpy as np
        write_results(args.output, records, {"scenarios": scenarios, "python": sys.version,
                                           "numpy": np.__version__, "opencv": cv2.__version__})
        print(f"Wrote {len(records)} decisions to {args.output}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, EOFError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
