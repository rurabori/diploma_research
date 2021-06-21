import json
import subprocess
from typing import List

AVAILABLE_ALGORITHMS = ['cpu_sequential', 'cpu_avx2', 'cuda']
INPUT_MATRICES = ['resources/matrices/barrier2-10/barrier2-10.mtx',
                  'resources/matrices/ns3Da/ns3Da.mtx',
                  'resources/matrices/west0156/west0156.mtx',
                  'resources/matrices/e40r5000.mtx'
                  ]
NUM_TRIES = 2

PARSED_OUTPUTS = ['SpMV', 'consumed_memory']


def get_stats(lines: List[bytes]):
    duration = 0.
    consumed_memory = 0
    for line in map(lambda line: line.decode('utf-8'), lines):
        if line.startswith('SpMV='):
            duration = float(line.split('=')[1][:-2])
        if line.startswith('consumed_memory='):
            consumed_memory = int(line.split('=')[1])
    return duration, consumed_memory


def single_run(matrix, algorithm: str):
    output = subprocess.check_output(
        ['./build/apps/conjugate_gradient/Release/conjugate_gradient', '-m', matrix, '-a', algorithm])
    return get_stats(output.splitlines())


results = {}
for input in INPUT_MATRICES:
    matrix_out = results.setdefault(input, {})
    algo_out = matrix_out.setdefault('algorithms', {})

    for algorithm in AVAILABLE_ALGORITHMS:
        runs = [*map(lambda _: single_run(input, algorithm), range(NUM_TRIES))]

        matrix_out['memory_consumption'] = runs[0][1]

        average_runtime = sum(map(lambda x: x[0], runs)) / NUM_TRIES
        algo_out[algorithm] = {'runtime': average_runtime,
                               'throughput': matrix_out['memory_consumption'] / average_runtime}


print(json.dumps(results))
