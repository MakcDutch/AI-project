"""Harmonic transport boundary. Image values are optical-axis depth, SI meters.

The Gazebo Ogre2 depth shader exports point.x (camera forward), not length(point).
No ray-range correction or learned-model scaling belongs here.
"""
from dataclasses import dataclass
import math
import threading
import time
import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    index: int
    timestamp: float  # original Gazebo simulation seconds
    received: float  # local monotonic seconds, never treated as capture time
    depth: np.ndarray


def decode_depth(message, *, float32_format=13):
    w, h, step = message.width, message.height, message.step
    if not 0 < w <= 4096 or not 0 < h <= 4096:
        raise ValueError('invalid depth dimensions')
    if message.pixel_format_type != float32_format or step < 4*w or step % 4:
        raise ValueError('expected native little-endian Gazebo R_FLOAT32 image')
    if len(message.data) != h*step:
        raise ValueError('depth byte length/stride mismatch')
    stamp = message.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nsec < 1_000_000_000:
        raise ValueError('invalid simulation timestamp')
    depth = np.ndarray((h, w), dtype='<f4', buffer=message.data, strides=(step, 4)).copy()
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth, stamp.sec + stamp.nsec*1e-9


class GazeboCamera:
    """Latest-frame buffer plus independent sim clock, pose and wall contact feed."""
    name = 'gazebo_harmonic_depth_z_m'

    def __init__(self, world='obstacle_world', model='x500_depth_0'):
        from gz.transport13 import Node
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.contacts_pb2 import Contacts
        self.node = Node()
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.fault = None
        self.count = 0
        self.sim_time = None
        self.clock_received = None
        self.pose = None
        self.pose_received = None
        self.collision = False
        self.collision_events = []
        self.contacts_received = None
        self.model = model
        self.topic = None
        self.node.subscribe(Clock, f'/world/{world}/clock', self._clock)
        self.node.subscribe(Pose_V, f'/world/{world}/pose/info', self._pose)
        self.node.subscribe(Contacts, '/wall_contacts', self._contact)

    def discover(self):
        from gz.msgs10.image_pb2 import Image
        topics = [t for t in self.node.topic_list()
                  if 'depth' in t.lower().rsplit('/sensor/',1)[-1]
                  and not t.endswith(('/points', '/camera_info'))]
        candidates = []
        for topic in topics:
            publishers, _ = self.node.topic_info(topic)
            if publishers and all(p.msg_type_name == 'gz.msgs.Image' for p in publishers):
                candidates.append(topic)
        if len(candidates) != 1:
            raise RuntimeError(f'expected exactly one depth Image topic; discovered {candidates}')
        self.topic = candidates[0]
        if not self.node.subscribe(Image, self.topic, self._image):
            self.topic = None
            raise RuntimeError('Gazebo Image subscription rejected')
        return self.topic

    def _image(self, message):
        try:
            depth, stamp = decode_depth(message)
            with self.lock:
                if self.frame and depth.shape != self.frame.depth.shape:
                    raise ValueError('depth dimensions changed during experiment')
                if self.frame and stamp <= self.frame.timestamp:
                    raise ValueError('depth timestamp reset or reordered; restart experiment')
                self.count += 1
                self.frame = DepthFrame(self.count, stamp, time.monotonic(), depth)
                self.error = None
        except Exception as exc:
            with self.lock:
                self.fault = str(exc)
                self.error = str(exc)
                self.frame = None

    def _clock(self, message):
        stamp = message.sim.sec + message.sim.nsec*1e-9
        with self.lock:
            if self.sim_time is not None and stamp < self.sim_time:
                self.fault = 'simulation reset; restart experiment'
                self.error = self.fault
                self.frame = None
            self.sim_time = stamp
            self.clock_received = time.monotonic()

    def _pose(self, message):
        for pose in message.pose:
            if pose.name == self.model:
                with self.lock:
                    self.pose = {'x': pose.position.x, 'y': pose.position.y, 'z': pose.position.z,
                                 'qw': pose.orientation.w, 'qx': pose.orientation.x,
                                 'qy': pose.orientation.y, 'qz': pose.orientation.z}
                    self.pose_received = time.monotonic()

    def _contact(self, message):
        with self.lock:
            self.contacts_received = time.monotonic()
            # This sensor belongs to the wall collision. Gazebo collision fields
            # are Entity messages (often IDs only), not strings. Any wall contact
            # is relevant; never discard an unnamed collision.
            self.collision |= bool(message.contact)
            for contact in message.contact:
                if len(self.collision_events)<1000:
                    self.collision_events.append(dict(
                        timestamp_s=message.header.stamp.sec+message.header.stamp.nsec*1e-9,
                        collision1={'id':contact.collision1.id,'name':contact.collision1.name},
                        collision2={'id':contact.collision2.id,'name':contact.collision2.name},
                        points=len(contact.position)))

    def snapshot(self):
        with self.lock:
            return dict(frame=self.frame, error=self.fault or self.error, sim_time=self.sim_time,
                        clock_received=self.clock_received, pose=self.pose,
                        pose_received=self.pose_received, collision=self.collision,
                        contacts_received=self.contacts_received, count=self.count,
                        collision_events=list(self.collision_events))

    def close(self):
        for topic in self.node.subscribed_topics():
            self.node.unsubscribe(topic)
