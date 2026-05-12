#include "common.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm>
static __global__ void reduce_rows_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[row * ncols + i];
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // sum up partial sums
    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}

template <bool norm>
static __global__ void reduce_rows_f32_strided(
        const float * __restrict__ x,
        float * __restrict__ dst,
        const int64_t ncols,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nb0,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {
    const int64_t row = blockIdx.x;
    const int64_t col = threadIdx.x;

    const int64_t i3 = row / (ne2*ne1);
    const int64_t ir = row - i3*ne2*ne1;
    const int64_t i2 = ir / ne1;
    const int64_t i1 = ir - i2*ne1;

    const int64_t base = i1*nb1 + i2*nb2 + i3*nb3;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };
    for (int64_t i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = x[base + i*nb0];
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? sum / ncols : sum;
}
