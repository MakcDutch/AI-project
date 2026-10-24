"""Actual wall-contact positive control in an isolated static-camera world.

No PX4 or flight commands. Must use the separate ai_project_contact_probe partition.
"""
import argparse
import json
import os
from pathlib import Path
import time
import subprocess


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if os.environ.get('GZ_PARTITION')!='ai_project_contact_probe':
        parser.error('requires isolated GZ_PARTITION=ai_project_contact_probe')
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.boolean_pb2 import Boolean
    from .gazebo_camera import GazeboCamera
    camera=GazeboCamera(model='depth_probe')
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and camera.snapshot()['pose'] is None:
            time.sleep(.05)
        if camera.snapshot()['pose'] is None:
            raise RuntimeError('isolated static depth_probe world not found; refusing insertion')
        deadline=time.monotonic()+5
        while '/world/obstacle_world/create' not in camera.node.service_list():
            if time.monotonic()>deadline:
                raise TimeoutError('Gazebo create service not discovered')
            time.sleep(.05)
        model_name=f'contact_validation_{time.monotonic_ns()}'
        sdf=f'''<sdf version="1.9"><model name="{model_name}">
          <pose>6 0 1.5 0 0 0</pose><link name="link">
          <collision name="cube"><geometry><box><size>0.3 0.3 0.3</size></box></geometry></collision>
          </link></model></sdf>'''
        # The installed synchronous Python request binding can starve subscribed
        # Python callbacks while waiting. Use Gazebo's native CLI for this service
        # call; subprocess wait releases Python's GIL so Contacts can be delivered.
        response=subprocess.run(['gz','service','-s','/world/obstacle_world/create',
            '--reqtype','gz.msgs.EntityFactory','--reptype','gz.msgs.Boolean',
            '--timeout','10000','--req',str(EntityFactory(sdf=sdf))],
            capture_output=True,text=True,timeout=15,check=True)
        if 'data: true' not in response.stdout:
            raise RuntimeError(f'Gazebo rejected insertion: {response.stdout} {response.stderr}')
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not camera.snapshot()['collision']:
            time.sleep(.05)
        s=camera.snapshot()
        result=dict(kind='actual Gazebo dynamic cube versus wall positive control; no aircraft',
                    success=any(model_name in e[side]['name'] for e in s['collision_events'] for side in ('collision1','collision2')),
                    model_name=model_name,collision_events=s['collision_events'],
                    simulation_time_s=s['sim_time'],probe_camera_pose=s['pose'])
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='collision_events'},indent=2))
        return 0 if result['success'] else 1
    finally:
        camera.close()


if __name__=='__main__':
    raise SystemExit(main())
