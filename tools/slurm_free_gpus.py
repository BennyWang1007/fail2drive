#!/usr/bin/env python3
"""Display unallocated whole GPUs in sinfo's default node order."""
import argparse
import re
import subprocess


def output(*args):
    return subprocess.check_output(args, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='also show nodes with no free GPUs')
    args = parser.parse_args()
    order = dict.fromkeys(output('sinfo', '-N', '-h', '-o', '%N').split())
    nodes = {}
    for line in output('scontrol', 'show', 'nodes', '-o').splitlines():
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if 'NodeName' in fields:
            nodes[fields['NodeName']] = fields
    used = {}
    for line in output('sinfo', '-N', '-h', '-O', 'NodeHost:80,GresUsed:2048').splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            used[parts[0]] = parts[1]

    print(f"{'NODE':<9} {'GPU_FREE/TOTAL':>14} {'CPU_FREE/TOTAL':>15} "
          f"{'RAM_FREE/TOTAL_GiB':>20} {'GPU_TYPE':<30} STATE")
    total_free = 0
    for name in order:
        node = nodes.get(name)
        if not node:
            continue
        cfg = dict(item.split('=', 1) for item in node.get('CfgTRES', '').split(',') if '=' in item)
        allocated = dict(item.split('=', 1) for item in node.get('AllocTRES', '').split(',') if '=' in item)
        total = int(cfg.get('gres/gpu', 0))
        occupied = int(allocated.get('gres/gpu', 0))
        # A partially shared physical GPU is unavailable as a whole GPU too.
        for payload in re.findall(r'(?:^|,)shard:[^:,]+:\d+\(([^)]*)\)', used.get(name, '')):
            occupied += sum(int(count) > 0 for count in re.findall(r'(\d+)/\d+', payload))
        free = max(0, total - occupied)
        if not free and not args.all:
            continue
        cpu_total = int(node.get('CPUEfctv', node['CPUTot']))
        cpu_free = max(0, cpu_total - int(node['CPUAlloc']))
        ram_total = int(node['RealMemory']) / 1024
        ram_free = max(0, int(node['RealMemory']) - int(node.get('AllocMem', 0))) / 1024
        models = dict.fromkeys(re.findall(r'(?:^|,)gpu:([^:,]+):\d+', node.get('Gres', '')))
        model = ','.join(s.removeprefix('nvidia_geforce_') for s in models) or '-'
        print(f"{name:<9} {str(free)+'/'+str(total):>14} {str(cpu_free)+'/'+str(cpu_total):>15} "
              f"{ram_free:.1f}/{ram_total:.1f}".ljust(62) + f" {model:<30} {node['State']}")
        total_free += free
    print(f'Unallocated whole GPUs shown: {total_free}')
    print('RAM is Slurm unallocated memory, not OS MemAvailable; check STATE/reservations before submitting.')


if __name__ == '__main__':
    main()
