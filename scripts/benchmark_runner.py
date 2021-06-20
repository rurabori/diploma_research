import argparse
from os import PathLike
import subprocess
from typing import List

AVAILABLE_ALGORITHMS = ['cpu_sequential', 'cpu_avx2', 'cuda']
INPUT_MATRICES = [  # 'resources/matrices/barrier2-10/barrier2-10.mtx',
    # 'resources/matrices/ns3Da/ns3Da.mtx',
    'resources/matrices/west0156/west0156.mtx']
NUM_TRIES = 5


def get_duration(lines: List[bytes]):
    decoded_lines = map(lambda line: line.decode('utf-8'), lines)
    return float(next(filter(lambda line: line.startswith('SpMV='), decoded_lines)).split('=')[1][:-2])


def single_run(matrix, algorithm: str):
    output = subprocess.check_output(
        ['./build/apps/conjugate_gradient/Release/conjugate_gradient', '-m', matrix, '-a', algorithm])
    return get_duration(output.splitlines())


results = {}
for input in INPUT_MATRICES:
    matrix_out = results.setdefault(input, {})
    for algorithm in AVAILABLE_ALGORITHMS:
        matrix_out[algorithm] = sum(map(lambda _: single_run(input, algorithm),
                                        range(NUM_TRIES))) / NUM_TRIES


print(results)
