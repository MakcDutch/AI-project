"""No real network inference: fake predict methods / fake neural backends only."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tello_nav.cli import main
from tello_nav.drone import MockDroneAdapter
from tello_nav.replay.model_provider import CameraCalibration, TeamModelDepthProvider, create_model_provider
from tello_nav.replay.providers import synthetic_frames
from tello_nav.replay.runner import ReplayConfig, ReplayRunner


class CalibrationFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cal_path = self.root / "fixture_camera.json"
        self.cal_path.write_text(json.dumps({"size": [90, 60],
            "K": [[90, 0, 45], [0, 100, 30], [0, 0, 1]], "dist": [0, 0, 0, 0, 0]}))
        self.calibration = CameraCalibration(self.cal_path)
        self.frame = next(synthetic_frames(1))

    def model(self, depth=4.0):
        class FakeModel:
            name = "fake-network-not-real-inference"

            def predict(inner, rgb, fx):
                inner.rgb, inner.fx = rgb, fx
                return np.full(rgb.shape[:2], depth, np.float32)
        return FakeModel()


class ModelIntegrationTests(CalibrationFixture, unittest.TestCase):
    def test_calibration_preserves_team_focal_convention(self):
        with patch("sys.dont_write_bytecode", True):
            from src.calibration.intrinsics import load_focal
        self.assertEqual(self.calibration.focal_for(180, 120), load_focal(self.cal_path, 180, 120))
        self.assertEqual(self.calibration.focal_for(90, 60), 95)
        for width, height in ((60, 90), (90, 90), (0, 60)):
            with self.assertRaises(ValueError):
                self.calibration.focal_for(width, height)

    def test_bad_calibration(self):
        for data in ({"size": [90, 60], "K": np.eye(2).tolist()},
                     {"size": [90, 0], "K": np.eye(3).tolist()},
                     {"size": [90, 60], "K": [[0, 0, 0], [0, 100, 0], [0, 0, 1]]}):
            self.cal_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                CameraCalibration(self.cal_path)

    def test_rgb_focal_alignment_and_no_double_scaling(self):
        self.frame.rgb[:, :, 0] = 201
        for kind in ("da3", "metric3d"):
            model = self.model()
            depth = TeamModelDepthProvider(model, self.calibration, kind=kind).predict(self.frame)
            self.assertEqual(model.fx, 95)
            np.testing.assert_array_equal(model.rgb, self.frame.rgb)
            self.assertEqual(model.rgb.dtype, np.uint8)
            self.assertEqual(depth.shape, (60, 90))
            self.assertTrue(np.all(depth == 4))
            model.rgb[:] = 0
            self.assertEqual(self.frame.rgb[0, 0, 0], 201)

    def test_invalid_and_saturated_output_is_unknown(self):
        for kind in ("da3", "metric3d"):
            for value in (0, -1, np.nan, np.inf, -np.inf):
                provider = TeamModelDepthProvider(self.model(value), self.calibration, kind=kind)
                rows = ReplayRunner(provider).run(synthetic_frames(2), "invalid")
                self.assertTrue(all(r["controller_state"] == "STOP" for r in rows))
        provider = TeamModelDepthProvider(self.model(300), self.calibration, kind="metric3d")
        self.assertTrue(np.isnan(provider.predict(self.frame)).all())

    def test_wrong_shape_and_bad_rgb_fail_closed(self):
        model = self.model()
        model.predict = lambda rgb, fx: np.ones((10, 10))
        provider = TeamModelDepthProvider(model, self.calibration, kind="da3")
        rows = ReplayRunner(provider).run(synthetic_frames(2), "shape")
        self.assertTrue(all(r["error"] and r["controller_state"] == "STOP" for r in rows))
        bad = type(self.frame)(0, 0, np.ones((60, 90, 3), float))
        with self.assertRaises(ValueError):
            provider.predict(bad)

    def test_download_gate_precedes_model_import(self):
        with patch("tello_nav.replay.model_provider.importlib.import_module") as imports:
            for kind in ("da3", "metric3d"):
                with self.assertRaises(PermissionError):
                    create_model_provider(kind, self.cal_path)
            imports.assert_not_called()

    def test_factory_uses_existing_constructor_signatures(self):
        for kind, attr in (("da3", "DA3Metric"), ("metric3d", "Metric3D")):
            module = types.SimpleNamespace()
            from unittest.mock import Mock
            constructor = Mock(return_value=self.model())
            setattr(module, attr, constructor)
            with patch("tello_nav.replay.model_provider.importlib.import_module", return_value=module):
                provider = create_model_provider(kind, self.cal_path, allow_model_download=True)
            self.assertEqual(provider.predict(self.frame).shape, (60, 90))
            if kind == "da3":
                constructor.assert_called_once_with(device="cpu")
            else:
                constructor.assert_called_once_with("metric3d_vit_small", device="cpu")

    def test_model_dependency_error_has_context(self):
        with patch("tello_nav.replay.model_provider.importlib.import_module", side_effect=ImportError("dependency missing")):
            with self.assertRaisesRegex(RuntimeError, "da3 initialization failed"):
                create_model_provider("da3", self.cal_path, allow_model_download=True)

    def test_mock_execution_and_failure_and_delay(self):
        provider = TeamModelDepthProvider(self.model(), self.calibration, kind="da3")
        runner = ReplayRunner(provider, mock_drone=True)
        a, b = runner.run(synthetic_frames(4), "pipeline"), runner.run(synthetic_frames(4), "pipeline")
        for rows in (a, b):
            for row in rows:
                row.pop("measured_inference_s")
        self.assertEqual(a, b)
        self.assertTrue(a[1]["mock_execution"]["accepted"])
        self.assertEqual(a[1]["mock_execution"]["effective_command"]["state"], "FORWARD")
        delayed = ReplayRunner(provider, config=ReplayConfig(mode="virtual", inference_delay_s=.6), mock_drone=True)
        row = delayed.run(synthetic_frames(1), "delay")[0]
        self.assertFalse(row["mock_execution"]["accepted"])
        self.assertEqual(row["mock_execution"]["effective_command"]["state"], "STOP")
        self.assertTrue(row["mock_execution"]["inhibited"])
        provider.model.predict = lambda rgb, fx: (_ for _ in ()).throw(RuntimeError("inference failed"))
        row = ReplayRunner(provider, mock_drone=True).run(synthetic_frames(1), "failure")[0]
        self.assertEqual(row["controller_state"], "STOP")
        self.assertIn("inference failed", row["error"])

    def test_out_of_order_adapter_command_is_rejected(self):
        from tello_nav.controller import ControlCommand
        adapter = MockDroneAdapter(clock=lambda: 1.2)
        adapter.connect()
        cmd = ControlCommand(state="FORWARD", forward_backward=10)
        self.assertTrue(adapter.send(cmd, timestamp=1.2, observation_timestamp=1.1))
        self.assertFalse(adapter.send(cmd, timestamp=1.15, observation_timestamp=1.05))
        self.assertTrue(adapter.inhibited)
        self.assertIn("out-of-order", adapter.last_error)

    def test_cli_models_manifest_and_mock_execution(self):
        video = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (90, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.zeros((60, 90, 3), np.uint8))
        writer.release()
        timestamps = self.root / "times.csv"
        timestamps.write_text("frame_index,timestamp_s\n0,100\n1,100.1\n2,100.2\n")
        manifest = self.root / "scenarios.json"
        manifest.write_text(json.dumps([{"id": kind, "video": video.name, "timestamps": timestamps.name,
                                       "depth_model": kind, "calibration": self.cal_path.name}
                                       for kind in ("da3", "metric3d")]))
        def provider(kind, path, **kwargs):
            self.assertEqual(Path(path), self.cal_path)
            return TeamModelDepthProvider(self.model(), CameraCalibration(path), kind=kind)
        output = self.root / "results.json"
        with patch("tello_nav.replay.model_provider.create_model_provider", side_effect=provider):
            self.assertEqual(main(["--manifest", str(manifest), "--mock-drone", "--output", str(output)]), 0)
        rows = json.loads(output.read_text())["records"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["frame_timestamp_s"], 100)
        self.assertTrue(rows[1]["mock_execution"]["accepted"])
        self.assertEqual(rows[4]["controller_state"], "FORWARD")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--video", str(video), "--depth-model", "da3", "--calibration", str(self.cal_path),
                      "--output", str(output)])
        self.assertEqual(error.exception.code, 2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch unavailable; wrapper math tests skipped")
class ExistingWrapperTransformTests(CalibrationFixture, unittest.TestCase):
    """Run teammates' actual predict bodies with fake network tensors, no weights."""

    def load_wrapper(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with patch("sys.dont_write_bytecode", True):
            spec.loader.exec_module(module)
        return module

    def test_da3_actual_predict_preprocess_and_scale(self):
        # Import native torch before patch.dict snapshots sys.modules. Restoring
        # that snapshot must not unload torch and reinitialize its C extension.
        import torch
        fake_api = types.ModuleType("depth_anything_3.api")
        fake_api.DepthAnything3 = object
        with patch.dict("sys.modules", {"depth_anything_3": types.ModuleType("depth_anything_3"),
                                        "depth_anything_3.api": fake_api}):
            module = self.load_wrapper("test_da3_source", "src/depth_models/da3.py")
        wrapper = module.DA3Metric.__new__(module.DA3Metric)
        received = []
        class FakeNetwork:
            def inference(inner, paths):
                received.append(cv2.imread(paths[0]))
                return types.SimpleNamespace(depth=np.full((1, 28, 42), 6, np.float32))
        wrapper.model = FakeNetwork()
        self.frame.rgb[:] = [200, 100, 25]
        provider = TeamModelDepthProvider(wrapper, self.calibration, kind="da3")
        depth = provider.predict(self.frame)
        np.testing.assert_array_equal(received[0][0, 0], [25, 100, 200])
        np.testing.assert_allclose(depth, (95 * 42 / 90) * 6 / 300, rtol=1e-5)
        self.assertEqual(depth.shape, (60, 90))
        row = ReplayRunner(provider).run([self.frame], "actual-wrapper-fake-network")[0]
        self.assertEqual(row["controller_state"], "BRAKE")

    def test_metric3d_actual_predict_preprocess_scale_and_inf_clamp(self):
        import torch
        module = self.load_wrapper("test_metric3d_source", "src/depth_models/metric3d.py")
        wrapper = module.Metric3D.__new__(module.Metric3D)
        wrapper.name = "metric3d_vit_small"
        wrapper.device = torch.device("cpu")
        wrapper.mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
        wrapper.std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
        received = []
        class FakeNetwork:
            value = 2.0
            def inference(inner, inputs):
                received.append(inputs["input"].shape)
                return torch.full((1, 1, 616, 1064), inner.value), None, {}
        wrapper.model = FakeNetwork()
        provider = TeamModelDepthProvider(wrapper, self.calibration, kind="metric3d")
        depth = provider.predict(self.frame)
        self.assertEqual(tuple(received[0]), (1, 3, 616, 1064))
        np.testing.assert_allclose(depth, 2 * (95 * (616 / 60) / 1000), rtol=1e-5)
        rows = ReplayRunner(provider, mock_drone=True).run(synthetic_frames(2), "actual-wrapper-fake-network")
        self.assertEqual(rows[1]["controller_state"], "FORWARD")
        wrapper.model.value = float("inf")
        self.assertTrue(np.isnan(provider.predict(self.frame)).all())
