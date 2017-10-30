import pytest
import numpy
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.algorithms.nn_index.hash_index.vptree import VPTreeHashIndex

_index_type_dict = {"BallTree": (SkLearnBallTreeHashIndex, {"random_seed": 0}),
                    "VPTree": (VPTreeHashIndex, {"random_seed": 0,
                                                 "tree_type": "vp"}),
                    "VPSTree": (VPTreeHashIndex, {"random_seed": 0,
                                                  "tree_type": "vps"}),
                    "VPSBTree": (VPTreeHashIndex, {"random_seed": 0,
                                                   "tree_type": "vpsb"})}

indices = sorted(_index_type_dict.keys())
sample_sizes = [10**3, 10**6, 10**7, 10**8]
dimensions = [4096]
random_seeds = [0]


@pytest.mark.parametrize('sample_size', sample_sizes)
@pytest.mark.parametrize('dimension', dimensions)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('index', indices)
@pytest.mark.benchmark(group="build_index")
def test_build(benchmark, sample_size, dimension, random_seed, index):
    index = _index_type_dict[index]
    index = index[0](**index[1])
    numpy.random.seed(random_seed)
    hashes = numpy.random.randint(0, high=2,
                                  size=(sample_size, dimension), dtype='bool')
    benchmark(index.build_index, hashes)


@pytest.mark.parametrize('sample_size', sample_sizes)
@pytest.mark.parametrize('dimension', dimensions)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('index', indices)
@pytest.mark.benchmark(group="nn_exact")
def test_nn_exact(benchmark, sample_size, dimension, random_seed, index):
    index = _index_type_dict[index]
    index = index[0](**index[1])
    numpy.random.seed(random_seed)
    hashes = numpy.random.randint(0, high=2,
                                  size=(sample_size, dimension), dtype='bool')
    query_vec = hashes[numpy.random.randint(hashes.shape[0])]
    index.build_index(hashes)
    result = benchmark(index.nn, query_vec)
    numpy.testing.assert_equal(result[0][0], query_vec)


@pytest.mark.parametrize('sample_size', sample_sizes)
@pytest.mark.parametrize('dimension', dimensions)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('index', indices)
@pytest.mark.benchmark(group="nn_random")
def test_nn_random(benchmark, sample_size, dimension, random_seed, index):
    index = _index_type_dict[index]
    index = index[0](**index[1])
    numpy.random.seed(random_seed)
    hashes = numpy.random.randint(0, high=2,
                                  size=(sample_size, dimension), dtype='bool')
    query_vec = numpy.random.randint(0, high=2, size=dimension,
                                     dtype='bool')
    index.build_index(hashes)
    benchmark(index.nn, query_vec)


@pytest.mark.parametrize('sample_size', sample_sizes)
@pytest.mark.parametrize('dimension', dimensions)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('index', indices)
@pytest.mark.benchmark(group="nn_near")
def test_nn_near(benchmark, sample_size, dimension, random_seed, index):
    index = _index_type_dict[index]
    index = index[0](**index[1])
    numpy.random.seed(random_seed)
    hashes = numpy.random.randint(0, high=2,
                                  size=(sample_size, dimension), dtype='bool')
    query_vec = hashes[numpy.random.randint(hashes.shape[0])]
    change_index = numpy.random.randint(query_vec.shape[0])
    query_vec[change_index] = not query_vec[change_index]
    index.build_index(hashes)
    result = benchmark(index.nn, query_vec)
    assert result[1][0] < 1.1 / query_vec.shape[0]
