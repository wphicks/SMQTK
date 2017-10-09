from __future__ import print_function
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
all_regex_strings = [
    r"^>>> Building ([A-Z]+)\.\.\.",
    r"^Generating (\d+) hashes of dimension (\d+)",
    r"^>>> ([A-Z]+) ([a-z]+ [a-z]+) test\.\.\.",
    r"^\s+[0-9]+ function calls .*in ([0-9]+\.[0-9]+) seconds",
    r"^\s+([0-9]+)/[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+.+\(.+knn_recursive_inner\)",
    r"^\s+[0-9]+\s+([0-9]+\.[0-9]+)\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+.+\(bit_vector_to_int_large\)",
    r"^\s+[0-9]+\s+([0-9]+\.[0-9]+)\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+.+\(int_to_bit_vector_large\)",
    r"^>>> ([A-Z]+) ([a-z]+ [a-z]+) test with ([0-9]+) changes\.\.\.",
]

all_regexes = [re.compile(s) for s in all_regex_strings]


def get_index_type(match, state):
    index_type = match.group(1)
    state[1][0] = index_type
    state[1][3] = "Build"


def get_config(match, state):
    num_hashes = int(match.group(1))
    dim = int(match.group(2))
    state[1][1] = dim
    state[1][2] = num_hashes


def get_test_type(match, state):
    state[1][0] = match.group(1)
    state[1][3] = match.group(2).title()


def get_runtime(match, state):
    state[1][4] = "Runtime"
    l = state[0]
    for spec in state[1][:-1]:
        l = l[spec]
    l[state[1][-1]].append(float(match.group(1)))


def get_vp_calls(match, state):
    state[1][4] = "VP Tree Search Calls"
    l = state[0]
    for spec in state[1][:-1]:
        l = l[spec]
    l[state[1][-1]].append(int(match.group(1)))


def get_bit2int_time(match, state):
    state[1][4] = "Bit to int conversion time"
    l = state[0]
    for spec in state[1][:-1]:
        l = l[spec]
    l[state[1][-1]].append(float(match.group(1)))


def get_int2bit_time(match, state):
    state[1][4] = "Int to bit conversion time"
    l = state[0]
    for spec in state[1][:-1]:
        l = l[spec]
    l[state[1][-1]].append(float(match.group(1)))


def get_near_changes(match, state):
    state[1][0] = match.group(1)
    state[1][3] = "{}_{}".format(
        match.group(2).title(), match.group(3)
    )


regex_callbacks = [
    get_index_type,
    get_config,
    get_test_type,
    get_runtime,
    get_vp_calls,
    get_bit2int_time,
    get_int2bit_time,
    get_near_changes,
]


def get_data(filename):
    specs = (None, None, None, None, None)
    dict_con = lambda: defaultdict(list)
    for i in xrange(len(specs) - 1):
        dict_con = lambda con=dict_con: defaultdict(con)
    run_data = (dict_con(), list(specs))

    with open(filename) as data_file:
        for line in data_file:
            results = [re.search(exp, line) for exp in all_regexes]
            if any(results):
                results = [
                    cb(m, run_data) for cb, m in zip(regex_callbacks, results)
                    if m
                ]
            else:
                results = []
    return run_data[0]


def build_graphs(data):
    graph_data = defaultdict(lambda: defaultdict(list))
    for index_type in data.keys():
        all_dim = sorted(data[index_type].keys())
        for dim in all_dim:
            all_N = sorted(data[index_type][dim].keys())
            for N in all_N:
                for run_type in data[index_type][dim][N].keys():
                    for data_type in data[index_type][dim][N][run_type].keys():
                        cur_data = data[
                            index_type][dim][N][run_type][data_type]
                        graph_spec = (run_type, data_type, N, "Dimension")
                        graph_data[graph_spec][index_type].extend(
                            [(dim, val) for val in cur_data])
                        graph_spec = (run_type, data_type, dim, "N")
                        graph_data[graph_spec][index_type].extend(
                            [(N, val) for val in cur_data])

    for spec in sorted(graph_data.keys(), key=lambda x: (x[3], x[2])):
        y_label = spec[1]
        if spec[-1] == "N":
            x_label = "Number of Hashes"
            title = "{} at dimension {}".format(spec[0], spec[2])
        elif spec[-1] == "Dimension":
            x_label = "Dimension"
            title = "{} with {} hashes".format(spec[0], spec[2])
        index_types = graph_data[spec].keys()
        legend_handles = []
        for ind_type in index_types:
            points = zip(*graph_data[spec][ind_type])
            handle = plt.plot(points[0], points[1], label=ind_type)[0]
            legend_handles.append(handle)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(handles=legend_handles)
        plt.title(title)
        plt.savefig("plots/{}-{}-{}-{}.png".format(*spec).replace(" ", "_"))
        plt.clf()


def print_tree(d, offset=0):
    try:
        keys = d.keys()
    except AttributeError:
        print("{}{}".format("| " * offset, d))
        return

    for key in keys:
        print("{}{}".format("| " * offset, key))
        print_tree(d[key], offset=offset + 1)


if __name__ == "__main__":
    parsed_data = get_data(sys.argv[1])
    build_graphs(parsed_data)
