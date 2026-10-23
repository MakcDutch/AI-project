"""Exchange fixtures only. No real inference, downloaded weights, or hardware."""
from dataclasses import replace
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.create_prediction_fixture import create_fixture
from tello_nav.cli import main
from tello_nav.replay.predictions import (
    ManifestDepthProvider, PredictionError, PredictionExporter, file_sha256,
)
from tello_nav.replay.runner import ReplayConfig, ReplayRunner
from tello_nav.replay.video import VideoReader


class PredictionExchangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture"
        create_fixture(self.root)
        self.manifest = self.root / "da3/predictions.json"
        self.video = self.root / "video.avi"
        self.calibration = self.root / "synthetic_calibration.json"
        self.frames = list(VideoReader(self.video, self.root / "timestamps.csv"))

    def load(self, path=None, **kwargs):
        options = dict(scenario_id="exchange-demo", video_path=self.video,
                       calibration_path=self.calibration, calibration_id="synthetic-camera-v1",
                       camera_id="synthetic-camera", allow_synthetic=True)
        options.update(kwargs)
        return ManifestDepthProvider(path or self.manifest, **options)

    def edit(self, callback):
        data = json.loads(self.manifest.read_text())
        callback(data)
        self.manifest.write_text(json.dumps(data))

    def exporter(self, target="new-export"):
        data = json.loads(self.manifest.read_text())
        return PredictionExporter(
            self.root / target, scenario_id="exchange-demo", frame_count=8, video_path=self.video,
            model=data["model"], calibration_path=self.calibration, calibration_id="synthetic-camera-v1",
            camera_id="synthetic-camera", calibration_status="synthetic", calibration_geometry="undistorted",
            depth=data["depth"])

    def add(self, exporter, frame=None, depth=None):
        row = json.loads(self.manifest.read_text())["frames"][0]
        exporter.add(frame or self.frames[0], depth if depth is not None else np.ones((60, 90), np.float32),
                     camera_intrinsics=row["camera_intrinsics"], inference_focal_px=90.0,
                     preprocessing=row["preprocessing"], inference_latency_s=.02)

    def test_two_model_exports_import_replay_and_mock_execution(self):
        for kind in ("da3", "metric3d"):
            provider = self.load(self.root / kind / "predictions.json")
            rows = ReplayRunner(provider, mock_drone=True).run(self.frames, kind)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["error"] is None for r in rows))
            self.assertTrue(all(r["mock_execution"]["accepted"] for r in rows))
            self.assertEqual(rows[1]["controller_state"], "FORWARD")
            self.assertEqual(rows[2]["controller_state"], "BRAKE")
            # Earlier braking/saturation may still inhibit recovery. A thin
            # obstacle must never permit forward translation in either state.
            self.assertIn(rows[6]["controller_state"], {"BRAKE", "STOP"})
            self.assertEqual(rows[6]["command"]["forward_backward"], 0)
            self.assertAlmostEqual(rows[6]["sectors"]["center"]["minimum_m"], .2, places=6)
            self.assertEqual(rows[7]["controller_state"], "STOP")
            self.assertEqual(rows[0]["prediction"]["frame_id"], "exchange-demo:000000")
            self.assertEqual(rows[0]["frame_timestamp_s"], 100)
            self.assertTrue(provider.metadata()["synthetic"])
            self.assertFalse(provider.metadata()["metric_evaluation_eligible_by_metadata"])

    def test_export_preserves_invalid_values_and_latency(self):
        exporter = self.exporter()
        depth = np.full((60, 90), 4.0, np.float32)
        depth[20, :5] = [np.nan, np.inf, -np.inf, 0, -2]
        for frame in self.frames:
            self.add(exporter, frame, depth)
        path = exporter.finalize()
        np.testing.assert_array_equal(np.load(path.parent / "depth/000000.npy"), depth)
        provider = self.load(path)
        np.testing.assert_array_equal(provider.predict(self.frames[0]), depth)
        record = ReplayRunner(provider).run(self.frames, "latency")[0]
        self.assertEqual(record["prediction"]["exported_inference_latency_s"], .02)

    def test_export_rejects_dtype_shape_and_duplicate(self):
        exporter = self.exporter()
        for depth in (np.ones((60, 90), np.float64), np.ones((10, 10), np.float32)):
            with self.assertRaises(ValueError):
                self.add(exporter, depth=depth)
        self.add(exporter)
        before = (exporter.root / "depth/000000.npy").read_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.add(exporter)
        self.assertEqual(before, (exporter.root / "depth/000000.npy").read_bytes())
        with self.assertRaises(FileExistsError):
            self.exporter()

    def test_incomplete_export_is_not_published(self):
        exporter = self.exporter()
        self.add(exporter)
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            exporter.finalize()
        self.assertFalse((exporter.root / "predictions.json").exists())

    def test_synthetic_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(PredictionError, "explicit opt-in"):
            self.load(allow_synthetic=False)

    def test_unknown_provisional_or_unverified_calibration_is_blocked(self):
        for status in ("provisional", "verified"):
            with self.subTest(status=status):
                self.edit(lambda d: d["calibration"].update(status=status))
                with self.assertRaisesRegex(PredictionError, "unsuitable calibration"):
                    self.load()
        self.edit(lambda d: d["model"].update(origin="pretrained"))
        self.edit(lambda d: d["calibration"].update(status="provisional"))
        with self.assertRaisesRegex(PredictionError, "unsuitable calibration"):
            self.load()

    def test_scenario_camera_id_and_calibration_id_checks(self):
        with self.assertRaisesRegex(PredictionError, "identifier mismatch"):
            self.load(self.root / "invalid/calibration_mismatch/predictions.json")
        for kwargs in ({"scenario_id": "wrong"}, {"camera_id": "tello"}, {"calibration_id": "wrong"}):
            with self.assertRaises(PredictionError):
                self.load(**kwargs)

    def test_calibration_and_video_hash_mismatch(self):
        alternate = self.root / "another_calibration.json"
        alternate.write_bytes(self.calibration.read_bytes() + b"\n")
        with self.assertRaisesRegex(PredictionError, "calibration SHA"):
            self.load(calibration_path=alternate)
        altered = self.root / "another.avi"
        altered.write_bytes(self.video.read_bytes() + b"different")
        with self.assertRaisesRegex(PredictionError, "video SHA"):
            self.load(video_path=altered)

    def test_missing_duplicate_or_invalid_frame_id(self):
        with self.assertRaises(PredictionError):
            self.load(self.root / "invalid/missing_prediction/predictions.json")
        original = self.manifest.read_text()
        for operation in (lambda d: d["frames"].pop(),
                          lambda d: d["frames"].append(d["frames"][0].copy()),
                          lambda d: d["frames"][0].update(frame_id="wrong:000000")):
            self.manifest.write_text(original)
            self.edit(operation)
            with self.assertRaises(PredictionError):
                self.load()

    def test_timestamp_mismatch_stops_then_requires_recovery(self):
        provider = self.load(self.root / "invalid/timestamp_mismatch/predictions.json")
        rows = ReplayRunner(provider, mock_drone=True).run(self.frames, "mismatch")
        self.assertEqual(rows[0]["controller_state"], "STOP")
        self.assertIn("timestamp mismatch", rows[0]["error"])
        self.assertEqual(rows[1]["controller_state"], "STOP")

    def test_timestamp_source_and_rgb_identity_checks(self):
        provider = self.load()
        for frame in (replace(self.frames[0], timestamp_source="fps_approximation"),
                      replace(self.frames[0], rgb=np.ones((60, 90, 3), np.uint8))):
            with self.assertRaises(PredictionError):
                provider.predict(frame)

    def test_nonmetric_units_and_radial_depth_rejected(self):
        with self.assertRaisesRegex(PredictionError, "require meters"):
            self.load(self.root / "invalid/radial_depth/predictions.json")
        original = self.manifest.read_text()
        for change in ({"units": "mm"}, {"units": "relative"}, {"convention": "radial_range"}):
            self.manifest.write_text(original)
            self.edit(lambda d: d["depth"].update(change))
            with self.assertRaisesRegex(PredictionError, "require meters"):
                self.load()

    def test_frame_geometry_and_intrinsics_rejected(self):
        original = self.manifest.read_text()
        for change in ({"depth_shape": [30, 45]}, {"camera_intrinsics": np.eye(3).tolist()},
                       {"image_shape": [90, 60], "depth_shape": [90, 60]}):
            self.manifest.write_text(original)
            self.edit(lambda d: d["frames"][0].update(change))
            with self.assertRaises(PredictionError):
                self.load()

    def test_corrupted_and_missing_files_fail_closed(self):
        provider = self.load(self.root / "invalid/corrupted_file/predictions.json")
        row = ReplayRunner(provider, mock_drone=True).run(self.frames, "corrupt")[0]
        self.assertEqual(row["controller_state"], "STOP")
        self.assertIn("SHA-256 mismatch", row["error"])
        provider = self.load()
        (self.root / "da3/depth/000000.npy").unlink()
        row = ReplayRunner(provider).run(self.frames, "missing")[0]
        self.assertEqual(row["controller_state"], "STOP")
        self.assertIn("cannot load depth", row["error"])

    def test_correct_hash_cannot_hide_bad_npy_dtype_or_shape(self):
        for array in (np.ones((60, 90), np.float64), np.ones((10, 10), np.float32)):
            path = self.root / "da3/depth/000000.npy"
            np.save(path, array)
            self.edit(lambda d: d["frames"][0].update(depth_sha256=file_sha256(path)))
            provider = self.load()
            with self.assertRaisesRegex(PredictionError, "dtype/shape"):
                provider.predict(self.frames[0])

    def test_correct_hash_cannot_hide_corrupted_npy(self):
        path = self.root / "da3/depth/000000.npy"
        path.write_bytes(b"not an npy")
        self.edit(lambda d: d["frames"][0].update(depth_sha256=file_sha256(path)))
        with self.assertRaises(PredictionError):
            self.load().predict(self.frames[0])

    def test_saturation_is_explicit_unknown(self):
        provider = self.load(self.root / "metric3d/predictions.json")
        self.assertTrue(np.isnan(provider.predict(self.frames[4])).all())
        original = np.load(self.root / "metric3d/depth/000004.npy")
        self.assertTrue(np.all(original == 300))
        rows = ReplayRunner(provider).run(self.frames, "saturation")
        self.assertEqual(rows[4]["controller_state"], "STOP")

    def test_path_traversal_and_symlink_rejected(self):
        self.edit(lambda d: d["frames"][0].update(depth_file="../outside.npy"))
        with self.assertRaisesRegex(PredictionError, "inside bundle"):
            self.load()
        external = self.root / "outside.npy"
        np.save(external, np.ones((60, 90), np.float32))
        link = self.root / "da3/link.npy"
        link.symlink_to(external)
        self.edit(lambda d: d["frames"][0].update(depth_file="link.npy"))
        with self.assertRaisesRegex(PredictionError, "escapes bundle"):
            self.load()

    def test_invalid_json_and_missing_essential_metadata(self):
        original = self.manifest.read_text()
        for value in ("{}", '{"schema_version":1,"schema_version":1}', "[1]"):
            self.manifest.write_text(value)
            with self.assertRaises(PredictionError):
                self.load()
        self.manifest.write_text(original)
        self.edit(lambda d: d["model"].pop("checkpoint_revision"))
        with self.assertRaises(PredictionError):
            self.load()

    def test_prediction_count_must_match_full_video(self):
        self.edit(lambda d: (d.update(frame_count=7), d["frames"].pop()))
        with self.assertRaisesRegex(PredictionError, "decoded video frame count"):
            self.load()

    def test_deterministic_replay_and_virtual_delays(self):
        provider = self.load()
        runner = ReplayRunner(provider, mock_drone=True)
        first, second = runner.run(self.frames, "demo"), runner.run(self.frames, "demo")
        for rows in (first, second):
            for row in rows:
                row.pop("measured_inference_s")
        self.assertEqual(first, second)
        runner = ReplayRunner(provider, config=ReplayConfig(mode="virtual", inference_delay_s=.6), mock_drone=True)
        rows = runner.run(self.frames, "delay")
        self.assertTrue(all(r["controller_state"] == "STOP" for r in rows))
        self.assertTrue(all(r["mock_execution"]["inhibited"] for r in rows))

    def test_step_by_step_import(self):
        iterator = ReplayRunner(self.load()).iter_records(self.frames, "step")
        self.assertEqual(next(iterator)["frame_index"], 0)
        self.assertEqual(next(iterator)["frame_index"], 1)
        iterator.close()

    def test_cli_multiple_models_and_logs(self):
        output = self.root / "log.json"
        import builtins
        from unittest.mock import patch
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.startswith(("torch", "depth_anything_3", "djitellopy", "src.depth_models")):
                raise AssertionError("external replay attempted neural/SDK import")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            self.assertEqual(main(["--manifest", str(self.root / "scenarios.json"),
                                  "--allow-synthetic-predictions", "--mock-drone", "--output", str(output)]), 0)
        data = json.loads(output.read_text())
        self.assertEqual(len(data["records"]), 16)
        self.assertEqual(data["metadata"]["scenarios"][1]["provider_metadata"]["model"]["kind"], "metric3d")
        self.assertTrue(all(r["prediction"]["source_scenario_id"] == "exchange-demo" for r in data["records"]))
        self.assertEqual(main(["--video", str(self.video), "--timestamps", str(self.root / "timestamps.csv"),
                              "--predictions", str(self.manifest), "--scenario-id", "exchange-demo",
                              "--calibration", str(self.calibration), "--calibration-id", "synthetic-camera-v1",
                              "--camera-id", "synthetic-camera", "--allow-synthetic-predictions",
                              "--output", str(self.root / "single.csv")]), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--manifest", str(self.root / "scenarios.json"), "--output", str(output)])
        self.assertEqual(error.exception.code, 2)
