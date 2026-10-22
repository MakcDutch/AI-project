import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tello_nav.cli import main
from tello_nav.controller import ControllerConfig
from tello_nav.controller.sectors import SectorConfig
from tello_nav.replay.logging import write_results
from tello_nav.replay.providers import CallableDepthProvider, MockDepthProvider, NpyDepthProvider, synthetic_frames
from tello_nav.replay.runner import ReplayConfig, ReplayRunner
from tello_nav.replay.video import Frame, VideoReader, read_timestamps
from tello_nav.replay.visualization import ReplayViewer


class ReplayTests(unittest.TestCase):
    def runner(self, scenario="empty_corridor", config=None):
        return ReplayRunner(MockDepthProvider(scenario), config=config)

    def test_all_synthetic_scenarios_and_determinism(self):
        for scenario in MockDepthProvider.SCENARIOS:
            runner = self.runner(scenario)
            a = runner.run(synthetic_frames(10), scenario)
            b = runner.run(synthetic_frames(10), scenario)
            for rows in (a, b):
                for row in rows:
                    row.pop("measured_inference_s")
            self.assertEqual(a, b)
            self.assertEqual(len(a), 10)
        rows = self.runner("reflective_surface").run(synthetic_frames(4), "glass")
        self.assertTrue(all(row["controller_state"] == "STOP" for row in rows))

    def test_step_iterator_is_lazy(self):
        calls = []
        provider = CallableDepthProvider("test", lambda f: calls.append(f.index) or np.ones((60, 90)) * 4)
        iterator = ReplayRunner(provider).iter_records(synthetic_frames(3), "test")
        self.assertEqual(calls, [])
        self.assertEqual(next(iterator)["frame_index"], 0)
        self.assertEqual(calls, [0])
        iterator.close()

    def test_decisions_do_not_modify_recorded_observations(self):
        frames = list(synthetic_frames(8))
        original = [f.rgb.copy() for f in frames]
        provider = MockDepthProvider("center_obstacle")
        safe = ReplayRunner(provider).run(frames, "fixed")
        lateral = ReplayRunner(provider, ControllerConfig(allow_lateral=True)).run(frames, "fixed")
        self.assertNotEqual([r["command"] for r in safe], [r["command"] for r in lateral])
        for before, after in zip(original, frames):
            np.testing.assert_array_equal(before, after.rgb)

    def test_visualization_renders_without_a_display(self):
        frame = next(synthetic_frames(1))
        provider = MockDepthProvider()
        row = ReplayRunner(provider).run([frame], "view")[0]
        viewer = ReplayViewer(SectorConfig())
        with patch("cv2.imshow") as show, patch("cv2.waitKey", return_value=ord("q")):
            self.assertFalse(viewer.show(frame, provider.predict(frame), row))
            canvas = show.call_args.args[1]
            self.assertEqual(canvas.shape[2], 3)
            self.assertGreater(canvas.shape[0], frame.rgb.shape[0])
            self.assertTrue(canvas.any())

    def test_npy_alignment_missing_and_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "000001.npy", np.ones((60, 90)) * 5)
            provider = NpyDepthProvider(tmp)
            rows = ReplayRunner(provider).run(synthetic_frames(3), "test")
            self.assertIn("FileNotFoundError", rows[0]["error"])
            self.assertIsNone(rows[1]["error"])
            self.assertEqual(rows[1]["sectors"]["center"]["distance_m"], 5)
            np.save(Path(tmp) / "000000.npy", np.ones((3, 4)))
            rows = ReplayRunner(provider).run(synthetic_frames(1), "shape")
            self.assertIn("aligned", rows[0]["error"])

    def test_inference_failure_stops_and_recovers(self):
        def infer(frame):
            if frame.index == 2:
                raise RuntimeError("inference failed")
            return np.ones((60, 90)) * 4
        rows = ReplayRunner(CallableDepthProvider("test", infer)).run(synthetic_frames(5), "failure")
        self.assertEqual([row["controller_state"] for row in rows], ["STOP", "FORWARD", "STOP", "STOP", "FORWARD"])
        self.assertIn("RuntimeError", rows[2]["error"])

    def test_virtual_inference_and_communication_age(self):
        cfg = ReplayConfig(mode="virtual", inference_delay_s=.4, communication_delay_s=.2)
        rows = self.runner(config=cfg).run(synthetic_frames(3), "delay")
        self.assertTrue(all(row["controller_state"] == "STOP" for row in rows))
        self.assertAlmostEqual(rows[0]["frame_age_s"], .6)
        self.assertAlmostEqual(rows[1]["frame_age_s"], 1.1)
        self.assertEqual(rows[0]["simulated_inference_s"], .4)
        again = self.runner(config=cfg).run(synthetic_frames(3), "delay")
        self.assertEqual([row["frame_age_s"] for row in rows], [row["frame_age_s"] for row in again])

    def test_late_frames_are_logged_as_stop(self):
        cfg = ReplayConfig(mode="virtual", inference_delay_s=1, late_policy="drop", max_lateness_s=.2)
        rows = self.runner(config=cfg).run(synthetic_frames(4), "drop")
        self.assertTrue(rows[1]["dropped"])
        self.assertEqual(rows[1]["controller_state"], "STOP")
        self.assertEqual(rows[1]["simulated_inference_s"], 0)

    def test_realtime_pacing_with_fake_clock(self):
        now = [100.0]
        sleeps = []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        def infer(frame):
            now[0] += .02
            return np.ones((60, 90)) * 4
        runner = ReplayRunner(CallableDepthProvider("test", infer),
                              config=ReplayConfig(mode="realtime", inference_delay_s=.03),
                              clock=lambda: now[0], sleep=sleep)
        rows = runner.run(synthetic_frames(3), "real")
        self.assertAlmostEqual(rows[0]["measured_inference_s"], .02)
        self.assertAlmostEqual(rows[0]["frame_age_s"], .05)
        self.assertAlmostEqual(rows[-1]["decision_time_s"], .25)
        self.assertTrue(sleeps)

    def test_realtime_slow_inference_causes_stop(self):
        now = [0.0]
        def infer(frame):
            now[0] += .8
            return np.ones((60, 90)) * 4
        runner = ReplayRunner(CallableDepthProvider("slow", infer), config=ReplayConfig(mode="realtime"),
                              clock=lambda: now[0], sleep=lambda s: None)
        rows = runner.run(synthetic_frames(2), "slow")
        self.assertTrue(all(r["controller_state"] == "STOP" for r in rows))

    def test_bad_sequence_and_invalid_config(self):
        frames = list(synthetic_frames(2))
        for bad in ([frames[0], frames[0]], [frames[1], frames[0]]):
            with self.assertRaises(ValueError):
                self.runner().run(bad, "bad")
        with self.assertRaises(ValueError):
            ReplayConfig(inference_delay_s=-1)
        with self.assertRaises(ValueError):
            self.runner(config=ReplayConfig(inference_delay_s=1))
        with self.assertRaises(ValueError):
            CallableDepthProvider("relative", lambda f: None, convention="relative")

    def test_json_and_csv_logs(self):
        runner = self.runner()
        records = runner.run(synthetic_frames(2), "log")
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ("json", "csv"):
                path = Path(tmp) / f"log.{suffix}"
                write_results(path, records, runner.metadata())
                self.assertTrue(path.is_file())
            data = json.loads((Path(tmp) / "log.json").read_text())
            self.assertEqual(data["metadata"]["schema_version"], 1)
            with (Path(tmp) / "log.csv").open() as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(json.loads(row["command"])["state"], "STOP")
            self.assertTrue((Path(tmp) / "log.metadata.json").exists())


class RecordedVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, name="input.avi", count=4):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (90, 60))
        self.assertTrue(writer.isOpened(), "OpenCV MJPG encoder required for fixture")
        for i in range(count):
            writer.write(np.full((60, 90, 3), (i * 30, 40, 200), np.uint8))
        writer.release()
        return path

    def test_video_indices_rgb_and_timestamp_sources(self):
        video = self.video()
        frames = list(VideoReader(video))
        self.assertEqual([f.index for f in frames], [0, 1, 2, 3])
        self.assertEqual(frames[1].timestamp, .1)
        self.assertEqual(frames[0].timestamp_source, "fps_approximation")
        self.assertGreater(frames[0].rgb[0, 0, 0], frames[0].rgb[0, 0, 2])
        stamps = self.root / "times.csv"
        stamps.write_text("frame_index,timestamp_s\n0,10\n1,10.15\n2,10.4\n3,10.6\n")
        external = list(VideoReader(video, stamps))
        self.assertEqual(external[2].timestamp, 10.4)
        self.assertEqual(external[2].timestamp_source, "external")
        rows = ReplayRunner(MockDepthProvider()).run(external, "external")
        self.assertAlmostEqual(rows[2]["decision_time_s"], .4)

    def test_timestamp_validation(self):
        stamps = self.root / "bad.csv"
        for text in ("", "timestamp\n0\n", "frame_index,timestamp_s\n1,0\n",
                     "frame_index,timestamp_s\n0,nan\n", "frame_index,timestamp_s\n0,1\n1,1\n"):
            stamps.write_text(text)
            with self.assertRaises(ValueError):
                read_timestamps(stamps)
        video = self.video()
        for count in (2, 5):
            stamps.write_text("frame_index,timestamp_s\n" + "".join(f"{i},{i*.1}\n" for i in range(count)))
            with self.assertRaises(ValueError):
                list(VideoReader(video, stamps))

    def test_missing_and_malformed_video(self):
        with self.assertRaises(FileNotFoundError):
            VideoReader(self.root / "missing.avi")
        bad = self.root / "bad.avi"
        bad.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            list(VideoReader(bad))

    def test_multiple_recorded_scenarios_cli(self):
        specs = []
        for name in ("corridor", "center"):
            self.video(name + ".avi")
            depth_dir = self.root / name
            depth_dir.mkdir()
            for i in range(4):
                depth = np.ones((60, 90)) * 4
                if name == "center":
                    depth[:, 30:60] = .5
                np.save(depth_dir / f"{i:06d}.npy", depth)
            specs.append({"id": name, "video": name + ".avi", "depth_dir": name, "model_name": "fixture"})
        manifest = self.root / "scenarios.json"
        manifest.write_text(json.dumps(specs))
        output = self.root / "results.json"
        self.assertEqual(main(["--manifest", str(manifest), "--output", str(output)]), 0)
        data = json.loads(output.read_text())
        self.assertEqual(len(data["records"]), 8)
        self.assertEqual(data["records"][1]["controller_state"], "FORWARD")
        self.assertEqual(data["records"][4]["controller_state"], "BRAKE")
        self.assertEqual(len(data["metadata"]["scenarios"][0]["inputs"]), 5)

    def test_cli_synthetic_headless(self):
        path = self.root / "synthetic.csv"
        self.assertEqual(main(["--synthetic", "center_obstacle", "--frames", "6", "--output", str(path)]), 0)
        self.assertTrue(path.exists())
