import pytest
import numpy
from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex
from smqtk.algorithms.nn_index.hash_index.vptree import VPTreeHashIndex

random_seed = 0
sample_size = 10**4
dimension = 4096
hashes = numpy.random.randint(
    0, high=2, size=(sample_size, dimension), dtype='bool')
btree = SkLearnBallTreeHashIndex(random_seed=random_seed)
vptree = VPTreeHashIndex(random_seed=random_seed, tree_type="vp")
vpstree = VPTreeHashIndex(random_seed=random_seed, tree_type="vps")
vpsbtree = VPTreeHashIndex(random_seed=random_seed, tree_type="vpsb")
btree.build_index(hashes)
vptree.build_index(hashes)
vpstree.build_index(hashes)
vpsbtree.build_index(hashes)

random_query_vec = numpy.random.randint(0, high=2, size=dimension,
                                        dtype='bool')
exact_query_vec = hashes[numpy.random.randint(hashes.shape[0])]
near_query_vec = hashes[numpy.random.randint(hashes.shape[0])]
change_index = numpy.random.randint(near_query_vec.shape[0])
near_query_vec[change_index] = not near_query_vec[change_index]


@pytest.mark.benchmark(group="build_index")
def test_bt_build(benchmark):
    benchmark(SkLearnBallTreeHashIndex(random_seed=random_seed).build_index,
              hashes)


@pytest.mark.benchmark(group="build_index")
def test_vp_build(benchmark):
    benchmark(VPTreeHashIndex(random_seed=random_seed,
              tree_type="vp").build_index, hashes)


@pytest.mark.benchmark(group="build_index")
def test_vps_build(benchmark):
    benchmark(VPTreeHashIndex(random_seed=random_seed,
              tree_type="vps").build_index, hashes)


@pytest.mark.benchmark(group="build_index")
def test_vpsb_build(benchmark):
    benchmark(VPTreeHashIndex(random_seed=random_seed,
              tree_type="vpsb").build_index, hashes)


@pytest.mark.benchmark(group="nn_random")
def test_bt_nn_random(benchmark):
    benchmark(btree.nn, random_query_vec)


@pytest.mark.benchmark(group="nn_random")
def test_vp_nn_random(benchmark):
    benchmark(vptree.nn, random_query_vec)


@pytest.mark.benchmark(group="nn_random")
def test_vps_nn_random(benchmark):
    benchmark(vpstree.nn, random_query_vec)


@pytest.mark.benchmark(group="nn_random")
def test_vpsb_nn_random(benchmark):
    benchmark(vpsbtree.nn, random_query_vec)


@pytest.mark.benchmark(group="nn_exact")
def test_bt_nn_exact(benchmark):
    result = benchmark(btree.nn, exact_query_vec)
    numpy.testing.assert_equal(result[0][0], exact_query_vec)


@pytest.mark.benchmark(group="nn_exact")
def test_vp_nn_exact(benchmark):
    result = benchmark(vptree.nn, exact_query_vec)
    numpy.testing.assert_equal(result[0][0], exact_query_vec)


@pytest.mark.benchmark(group="nn_exact")
def test_vps_nn_exact(benchmark):
    result = benchmark(vpstree.nn, exact_query_vec)
    numpy.testing.assert_equal(result[0][0], exact_query_vec)


@pytest.mark.benchmark(group="nn_exact")
def test_vpsb_nn_exact(benchmark):
    result = benchmark(vpsbtree.nn, exact_query_vec)
    numpy.testing.assert_equal(result[0][0], exact_query_vec)


@pytest.mark.benchmark(group="nn_near")
def test_bt_nn_near(benchmark):
    result = benchmark(btree.nn, near_query_vec)
    assert result[1][0] < 1.1 / near_query_vec.shape[0]


@pytest.mark.benchmark(group="nn_near")
def test_vp_nn_near(benchmark):
    result = benchmark(vptree.nn, near_query_vec)
    assert result[1][0] < 1.1 / near_query_vec.shape[0]


@pytest.mark.benchmark(group="nn_near")
def test_vps_nn_near(benchmark):
    result = benchmark(vpstree.nn, near_query_vec)
    assert result[1][0] < 1.1 / near_query_vec.shape[0]


@pytest.mark.benchmark(group="nn_near")
def test_vpsb_nn_near(benchmark):
    result = benchmark(vpsbtree.nn, near_query_vec)
    assert result[1][0] < 1.1 / near_query_vec.shape[0]
