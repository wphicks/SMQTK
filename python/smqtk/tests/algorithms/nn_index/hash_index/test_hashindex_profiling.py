from __future__ import print_function
import cProfile

from six.moves import range
import numpy
import matplotlib.pyplot as plt
import pylab

from smqtk.algorithms.nn_index.hash_index.linear import LinearHashIndex
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.algorithms.nn_index.hash_index.vptree import VPTreeHashIndex


def _build_random_hashes(dimension, sample_size, random_seed=None):
    numpy.random.seed(random_seed)
    return numpy.random.randint(
        0, high=2, size=(sample_size, dimension), dtype='bool')


def profile_decorator(func):
    def profiled(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats()
        return result
    return profiled


@profile_decorator
def build_index(index, hashes):
    index.build_index(hashes)


@profile_decorator
def neighbors_test(index, hashes):
    return [index.nn(vec, n=1) for vec in hashes]


def check_exact_neighbors(hashes, neighbors):
    for i in range(len(hashes)):
        try:
            assert(numpy.array_equal(hashes[i], neighbors[i][0]))
        except AssertionError as exc:
            raise exc


def check_near_neighbors(dists, changes, dimension):
    for dist_ in dists:
        try:
            assert(dist_[0] < (changes + 0.5) / float(dimension))
        except AssertionError as exc:
            print(dist_[0], (changes + 0.5) / float(dimension))
            raise exc


def _hamming_distance(query, hash_):
    return sum(x != y for x, y in zip(query, hash_)) / float(len(hash_))


def _dist_to_nearest_neigbor(query, hashes):
    best = 1
    for h in hashes:
        new_dist = _hamming_distance(query, h)
        if new_dist < best:
            best = new_dist
    return best


def check_random_queries(dists, queries, hashes):
    dimension = len(hashes[0])
    for dist_, query in zip(dists, queries):
        try:
            assert(
                dist_[0] < _dist_to_nearest_neigbor(query, hashes) +
                0.5 / float(dimension)
            )
        except AssertionError as exc:
            print(dist_[0], _dist_to_nearest_neigbor(query, hashes))
            raise exc


def cumulative_gains(query, hashes, index):
    neighbors, dists = index.nn(query, 9)
    actual_neighbors = sorted(
        hashes, key=lambda h: _hamming_distance(query, h)
    )
    x_points = []
    y_points = []
    actual_rank = 0
    for an in actual_neighbors:
        x_points.append(actual_rank)
        for i in range(len(neighbors)):
            if (an == neighbors[i]).all():
                y_points.append(i)
                break
        if len(x_points) != len(y_points):
            y_points.append(len(neighbors))
        actual_rank += 1
    plt.plot(x_points, y_points)
    pylab.show()
    plt.clf()


def _alter_hash(vec, changes=1):
    vec = vec.copy()
    indices = numpy.random.choice(numpy.arange(0, len(vec)), changes)
    for index in indices:
        vec[index] = not vec[index]
    return vec


def generate_near_queries(hashes, changes=1):
    return [_alter_hash(h, changes=changes) for h in hashes]


def do_profiling(
        dimension=12, sample_size=100, random_seed=None, near_changes=2):
    print(
        "Generating {} hashes of dimension {} with random_seed {}".format(
            sample_size, dimension, random_seed
        )
    )
    hashes = _build_random_hashes(
        dimension, sample_size, random_seed=random_seed)
    near_queries = generate_near_queries(hashes, changes=near_changes)
    random_queries = _build_random_hashes(
        dimension, sample_size, random_seed=random_seed)

    indices = (
        ("LIN", LinearHashIndex()),
        ("BT", SkLearnBallTreeHashIndex(random_seed=random_seed)),
        ("VP", VPTreeHashIndex(random_seed=random_seed, tree_type="vp")),
        ("VPS", VPTreeHashIndex(random_seed=random_seed, tree_type="vps")),
        ("VPSB", VPTreeHashIndex(random_seed=random_seed, tree_type="vpsb")),
    )

    for name, index in indices:
        print(">>> Building {}...".format(name))
        build_index(index, hashes)

        print(">>> {} exact neighbors test...".format(name))
        results = neighbors_test(index, hashes)
        neighbors = [result_[0] for result_ in results]
        dists = [result_[1] for result_ in results]
        if sample_size <= 100:
            check_exact_neighbors(hashes, neighbors)

        print(">>> {} near neighbors test with {} changes...".format(
            name, near_changes)
        )
        results = neighbors_test(index, near_queries)
        neighbors = [result_[0] for result_ in results]
        dists = [result_[1] for result_ in results]
        if sample_size <= 100:
            check_near_neighbors(dists, near_changes, dimension)

        print(">>> {} random neighbors test...".format(name))
        results = neighbors_test(index, random_queries)
        neighbors = [result_[0] for result_ in results]
        dists = [result_[1] for result_ in results]
        if sample_size <= 100:
            check_random_queries(dists, random_queries, hashes)


if __name__ == "__main__":
    all_dim = (16, 128, 2048, 4096)
    all_sizes = (10, 100, 1000, 10000)
    all_changes = (1, 2, 3, 4)
    for sample_size in all_sizes:
        for dim in all_dim:
            for changes in all_changes:
                do_profiling(
                    dimension=dim, sample_size=sample_size,
                    random_seed=changes, near_changes=changes
                )
