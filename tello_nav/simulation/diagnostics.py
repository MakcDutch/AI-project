"""Reproducible real Gazebo static-camera probe, without PX4 or flight commands."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np


def make_world(path):
    source = Path(__file__).resolve().parents[2]/'simulation/worlds/obstacle_world.sdf'
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    model = ET.SubElement(world,'model',name='depth_probe')
    ET.SubElement(model,'static').text='true'
    ET.SubElement(model,'pose').text='0 0 1.5 0 0 0'
    link = ET.SubElement(model,'link',name='camera_link')
    sensor = ET.SubElement(link,'sensor',name='depth_probe',type='depth_camera')
    for key,value in [('topic','depth_camera'),('always_on','true'),('update_rate','10')]:
        ET.SubElement(sensor,key).text=value
    camera = ET.SubElement(sensor,'camera')
    ET.SubElement(camera,'horizontal_fov').text='1.274'
    image = ET.SubElement(camera,'image')
    for key,value in [('width','160'),('height','120'),('format','R_FLOAT32')]:
        ET.SubElement(image,key).text=value
    clip = ET.SubElement(camera,'clip')
    ET.SubElement(clip,'near').text='.2'
    ET.SubElement(clip,'far').text='19.1'
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    tree.write(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-probe-world')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.make_probe_world:
        make_world(args.make_probe_world)
        return
    if not args.output:
        parser.error('provide --make-probe-world or --output')
    from .gazebo_camera import GazeboCamera
    camera = GazeboCamera(model='depth_probe')
    samples=[]
    last_index=None
    deadline=time.monotonic()+25
    try:
        while time.monotonic()<deadline and len(samples)<20:
            if camera.topic is None:
                try: camera.discover()
                except RuntimeError: pass
            s=camera.snapshot()
            frame=s['frame']
            if frame is not None and frame.index!=last_index and s['sim_time'] is not None:
                d=frame.depth
                samples.append(dict(index=frame.index, frame_timestamp_s=frame.timestamp,
                    received_monotonic_s=frame.received, simulation_time_s=s['sim_time'],
                    shape=list(d.shape), center_z_m=float(d[d.shape[0]//2,d.shape[1]//2]),
                    finite_fraction=float(np.isfinite(d).mean())))
                last_index=frame.index
            time.sleep(.02)
        pubs,_=camera.node.topic_info('/wall_contacts')
        contact_available=any(p.msg_type_name=='gz.msgs.Contacts' for p in pubs)
        success=(len(samples)==20 and all(abs(r['center_z_m']-6)<.02 for r in samples)
                 and contact_available and s['pose'] is not None)
        result=dict(kind='actual Gazebo static camera probe; no aircraft or PX4',success=success,
            topic=camera.topic, camera_pose=s['pose'], contact_publisher_available=contact_available,
            sensor_rate_hz=(len(samples)-1)/(samples[-1]['frame_timestamp_s']-samples[0]['frame_timestamp_s']) if len(samples)>1 else None,
            receive_rate_hz=(len(samples)-1)/(samples[-1]['received_monotonic_s']-samples[0]['received_monotonic_s']) if len(samples)>1 else None,
            samples=samples,error=s['error'])
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
        if not success:
            raise SystemExit(1)
    finally:
        camera.close()


if __name__=='__main__':
    main()
