"""Read-only environment/PX4 checks. Does not arm, connect or install anything."""
import argparse
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--px4-root',required=True)
    parser.add_argument('--print-pid',action='store_true')
    args=parser.parse_args(argv)
    root=Path(args.px4_root).resolve()
    binary=root/'build/px4_sitl_default/bin/px4'
    processes=[]
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if (entry/'exe').resolve(strict=True)==binary.resolve():
                    processes.append(int(entry.name))
            except (OSError,RuntimeError):
                pass
    if args.print_pid:
        if len(processes)!=1:
            parser.error(f'expected one local SITL process, found {processes}')
        print(processes[0])
        return 0
    result=dict(px4_root=str(root), gz=shutil.which('gz'),cmake=shutil.which('cmake'),
        ninja=shutil.which('ninja'),
        x500_airframe=(root/'ROMFS/px4fmu_common/init.d-posix/airframes/4002_gz_x500_depth').is_file(),
        x500_depth_model=(root/'Tools/simulation/gz/models/x500_depth/model.sdf').is_file(),
        sitl_binary=binary.is_file(),local_sitl_pids=processes,
        python_packages={name:importlib.util.find_spec(name) is not None for name in ['mavsdk','numpy','matplotlib']},
        opencv_cpp_config=bool(list(Path('/usr').glob('lib/*/cmake/opencv4/OpenCVConfig.cmake'))))
    try:
        import gz.transport13
        import gz.msgs10.image_pb2
        result['gazebo_python_bindings']=True
    except ImportError:
        result['gazebo_python_bindings']=False
    try:
        result['mavsdk_version']=version('mavsdk')
    except Exception:
        result['mavsdk_version']=None
    print(json.dumps(result,indent=2))
    return 0 if all([result['x500_airframe'],result['x500_depth_model'],result['sitl_binary'],
                     result['gazebo_python_bindings'],*result['python_packages'].values()]) else 1


if __name__=='__main__':
    raise SystemExit(main())
