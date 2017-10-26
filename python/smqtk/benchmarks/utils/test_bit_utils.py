import numpy

from smqtk.utils.bit_utils import (
    bit_vector_to_int,
    int_to_bit_vector,
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)

small_bit_vec = numpy.random.randint(0, high=2, size=63, dtype='bool')
large_bit_vec = numpy.random.randint(0, high=2, size=4096, dtype='bool')
small_bit_vec_rev = small_bit_vec[::-1]
large_bit_vec_rev = large_bit_vec[::-1]

def old_bit_vector_to_int_large(v):
    c = 0L
    for b in v:
        c = (c * 2L) + int(b)
    return c

small_int = old_bit_vector_to_int_large(small_bit_vec)
large_int = old_bit_vector_to_int_large(large_bit_vec)
small_int_rev = old_bit_vector_to_int_large(small_bit_vec_rev)
large_int_rev = old_bit_vector_to_int_large(large_bit_vec_rev)

def test_bit_vector_to_int(benchmark):
    result = benchmark(bit_vector_to_int, small_bit_vec)
    assert result == small_int

def test_bit_vector_to_int_rev(benchmark):
    result = benchmark(bit_vector_to_int, small_bit_vec_rev)
    assert result == small_int_rev

def test_bit_vector_to_int_large(benchmark):
    result = benchmark(bit_vector_to_int_large, large_bit_vec)
    assert result == large_int

def test_bit_vector_to_int_large_rev(benchmark):
    result = benchmark(bit_vector_to_int_large, large_bit_vec_rev)
    assert result == large_int_rev

def test_int_to_bit_vector(benchmark):
    result = benchmark(int_to_bit_vector, small_int, bits=63)
    numpy.testing.assert_equal(result, small_bit_vec)

def test_int_to_bit_vector_large(benchmark):
    result = benchmark(int_to_bit_vector_large, large_int, bits=4096)
    numpy.testing.assert_equal(result, large_bit_vec)
