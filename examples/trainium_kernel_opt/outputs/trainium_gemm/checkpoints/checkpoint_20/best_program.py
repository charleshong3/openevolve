"""
Initial GEMM kernel for OpenEvolve optimization.

This kernel implements matrix multiplication using NKI (Neuron Kernel Interface)
for AWS Trainium hardware. OpenEvolve will evolve the code within the EVOLVE-BLOCK
markers to optimize performance while maintaining correctness.
"""

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np


# EVOLVE-BLOCK-START
@nki.jit
def test(lhsT, rhs):
    """NKI kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples of
          128.  It is the left-hand-side argument of the matrix multiplication,
          delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
          is a multiple of 512.  It is the right-hand-side argument of the matrix
          multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Precompute index tiles to avoid rematerialization and ensure P-first indexing
    i_km = nl.mgrid[0:TILE_K, 0:TILE_M]
    i_kn = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_mn = nl.mgrid[0:TILE_M, 0:TILE_N]

    m_tiles = M // TILE_M
    n_tiles = N // TILE_N
    k_tiles = K // TILE_K

    # Use affine_range to loop over tiles
    for m in nl.affine_range(m_tiles):
        for n in nl.affine_range(n_tiles):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(k_tiles):
                # Load tiles directly to SBUF and multiply on Tensor Engine
                lhs_km = nl.load(lhsT[k * TILE_K + i_km.p,
                                      m * TILE_M + i_km.x])
                rhs_kn = nl.load(rhs[k * TILE_K + i_kn.p,
                                     n * TILE_N + i_kn.x])
                # Use nc_matmul: stationary[K,M] and moving[K,N] → PSUM[M,N]
                res_psum += nisa.nc_matmul(lhs_km, rhs_kn)

            # Store result (implicit PSUM->SBUF copy performed by nl.store)
            nl.store(result[m * TILE_M + i_mn.p, n * TILE_N + i_mn.x],
                     value=res_psum)

    return result
# EVOLVE-BLOCK-END


# Reference implementation for correctness testing (not evolved)
@nki.jit
def nki_matmul_reference(lhsT, rhs):
    """Reference NKI matmul implementation with blocking for better memory access.

    This is used as the reference for correctness testing.
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Define the indices (shape) of the tiles
    i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    # Configuring the blocking size for the free dimensions
    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

    # Loop over blocks over the M dimension
    for m in nl.affine_range(M // BLOCK_M):
        # Load TILES_IN_BLOCK_M columns tiles from lhsT
        lhsT_tiles = nl.ndarray(
            (TILES_IN_BLOCK_M, K // TILE_K, nl.par_dim(TILE_K), TILE_M),
            dtype=lhsT.dtype,
            buffer=nl.sbuf)
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            for k in nl.affine_range(K // TILE_K):
                lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[k * TILE_K + i_lhsT.p,
                         (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x])

        for n in nl.affine_range(N // BLOCK_N):
            # Load TILES_IN_BLOCK_N columns from rhs
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf)
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
                        rhs[k * TILE_K + i_rhs.p,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x])

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    # Allocate a tensor in PSUM
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        # Accumulate partial-sums into PSUM
                        res_psum += nl.matmul(lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x],
                                              rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                                              transpose_x=True)

                    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                    res_sb = nl.copy(res_psum, dtype=result.dtype)
                    nl.store(result[(m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p,
                                    (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x],
                             value=res_sb)

    return result
