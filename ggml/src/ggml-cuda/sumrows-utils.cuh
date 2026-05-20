#pragma once

class ggml_cuda_sum_rows_utils {
public:
    static void launch_f32_maybe_strided(
            const ggml_tensor * src0,
            const float * src0_d,
            float * dst_d,
            const int64_t ncols,
            const ggml_cuda_kernel_launch_params & launch_params) {
        if (ggml_is_contiguous(src0)) {
            ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
            return;
        }

        ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/false>,
                launch_params,
                src0_d, dst_d, ncols, src0->ne[1], src0->ne[2],
                src0->nb[0]/sizeof(float), src0->nb[1]/sizeof(float), src0->nb[2]/sizeof(float), src0->nb[3]/sizeof(float));
    }
};
