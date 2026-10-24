"""Plot measured experiment logs, never synthesize flight measurements."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(Path(args.log).read_text())
    rows = data['records']
    if not rows:
        parser.error('log has no flight measurements; cannot plot a successful flight')
    t = [r['elapsed_s'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0,0]
    ax.plot(t, [r['wall_clearance_m'] for r in rows], label='Body clearance (Gazebo pose)')
    ax.plot(t, [r['center_depth_m'] for r in rows], label='Camera center Z')
    ax.set(xlabel='Simulation time since hover (s)', ylabel='Distance (m)'); ax.legend()
    ax = axes[0,1]
    ax.plot(t, [r['desired_velocity']['forward_m_s'] for r in rows], label='Controller request')
    ax.plot(t, [r['applied_velocity']['forward_m_s'] for r in rows], label='Applied setpoint')
    ax.plot(t, [r['actual_forward_m_s'] for r in rows], label='Measured body forward')
    ax.set(xlabel='Simulation time (s)', ylabel='Velocity (m/s)'); ax.legend()
    ax = axes[1,0]
    states = ['STOP','FORWARD','BRAKE','AVOID_LEFT','AVOID_RIGHT']
    ax.step(t, [states.index(r['controller_state']) for r in rows], where='post')
    ax.set_yticks(range(len(states)), states)
    ax.set(xlabel='Simulation time (s)', ylabel='Controller state')
    ax = axes[1,1]
    ax.plot([r['position']['x'] for r in rows], [r['position']['y'] for r in rows])
    ax.axvline(6, color='red', label='Wall front face')
    ax.set(xlabel='Gazebo ENU X / east (m)', ylabel='Gazebo ENU Y / north (m)'); ax.legend()
    fig.suptitle(f"{data['metadata']['experiment']}: success={data['metadata']['summary']['success']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
