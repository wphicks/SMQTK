from __future__ import print_function
import sys
import json
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_data(data):
    parsed_data = defaultdict(lambda: defaultdict(list))
    for benchmark in data["benchmarks"]:
        benchmark_name = "{}-{}".format(benchmark["group"],
                                        benchmark['params']['dimension'])
        parsed_data[benchmark_name][benchmark['params']['index']].append(
            (benchmark['params']['sample_size'], benchmark['stats']['median']))
    return parsed_data


def gen_plots(data):
    for test_name in data.keys():
        index_types = sorted(data[test_name])
        legend_handles = []
        for ind_type in index_types:
            points = zip(*data[test_name][ind_type])
            legend_handles.append(plt.plot(points[0], points[1],
                                  label=ind_type)[0])
        plt.title(test_name)
        plt.xlabel("Stored Hashes")
        plt.ylabel("Median Runtime")
        plt.legend(handles=legend_handles)
        plt.savefig("{}.png".format(test_name))
        plt.clf()


if __name__ == "__main__":
    with open(sys.argv[1]) as benchmark_file:
        data = json.load(benchmark_file)
    gen_plots(parse_data(data))
