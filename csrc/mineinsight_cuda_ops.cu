/**
 * MineInsight Custom CUDA Kernels
 *
 * Module: DEF-mineinsight
 * Target: L4 (sm_89), CUDA 12.8, PyTorch cu128
 *
 * 1. fused_multimodal_preprocess — RGB+LWIR/SWIR → normalized CHW in one pass
 * 2. fused_ciou_loss — Complete IoU loss on CUDA (eliminates 15+ elementwise ops)
 * 3. fused_detection_decode — Decode raw predictions to (boxes, scores, labels)
 *
 * Compile:
 *   cd /mnt/forge-data/modules/05_wave9/21_MineInsight
 *   source .venv/bin/activate
 *   python csrc/setup.py build_ext --inplace
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>


// ============================================================================
// Kernel 1: Fused multi-modal image preprocessing
// ============================================================================
// Takes two HWC uint8 images (RGB + thermal) and produces a single
// (2*C, H, W) normalized float32 tensor in one CUDA kernel pass.
// Eliminates: 2x permute + 2x float() + 2x /255 + 1x cat = 7 ops → 1 op.

__global__ void fused_multimodal_preprocess_kernel(
    const unsigned char* __restrict__ rgb,    // (H, W, 3) uint8
    const unsigned char* __restrict__ thermal, // (H, W, 3) uint8
    float* __restrict__ output,               // (6, H, W) float32
    int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W * 3;
    if (idx >= total) return;

    int c = idx % 3;
    int w = (idx / 3) % W;
    int h = idx / (3 * W);

    // RGB channels → first 3 channels of output (CHW layout)
    float rgb_val = (float)rgb[h * W * 3 + w * 3 + c] / 255.0f;
    output[c * H * W + h * W + w] = rgb_val;

    // Thermal channels → channels 3-5 of output (CHW layout)
    float therm_val = (float)thermal[h * W * 3 + w * 3 + c] / 255.0f;
    output[(3 + c) * H * W + h * W + w] = therm_val;
}


/**
 * Batch version: (B, H, W, 3) + (B, H, W, 3) → (B, 6, H, W).
 */
__global__ void fused_batch_multimodal_preprocess_kernel(
    const unsigned char* __restrict__ rgb,
    const unsigned char* __restrict__ thermal,
    float* __restrict__ output,
    int B, int H, int W
) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels_per_img = H * W * 3;
    int total = B * pixels_per_img;
    if (gidx >= total) return;

    int b = gidx / pixels_per_img;
    int rem = gidx % pixels_per_img;
    int c = rem % 3;
    int w = (rem / 3) % W;
    int h = rem / (3 * W);

    int in_offset = b * pixels_per_img;
    int out_offset = b * 6 * H * W;

    float rgb_val = (float)rgb[in_offset + h * W * 3 + w * 3 + c] / 255.0f;
    output[out_offset + c * H * W + h * W + w] = rgb_val;

    float therm_val = (float)thermal[in_offset + h * W * 3 + w * 3 + c] / 255.0f;
    output[out_offset + (3 + c) * H * W + h * W + w] = therm_val;
}


// ============================================================================
// Kernel 2: Fused CIoU loss computation
// ============================================================================
// Computes Complete IoU loss for N box pairs in parallel.
// Eliminates 15+ elementwise torch operations → 1 kernel.

__global__ void fused_ciou_loss_kernel(
    const float* __restrict__ pred,    // (N, 4) cx, cy, w, h
    const float* __restrict__ target,  // (N, 4) cx, cy, w, h
    float* __restrict__ losses,        // (N,) per-pair CIoU loss
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float pcx = pred[i * 4 + 0], pcy = pred[i * 4 + 1];
    float pw  = pred[i * 4 + 2], ph  = pred[i * 4 + 3];
    float tcx = target[i * 4 + 0], tcy = target[i * 4 + 1];
    float tw  = target[i * 4 + 2], th  = target[i * 4 + 3];

    // Convert to xyxy
    float px1 = pcx - pw * 0.5f, py1 = pcy - ph * 0.5f;
    float px2 = pcx + pw * 0.5f, py2 = pcy + ph * 0.5f;
    float tx1 = tcx - tw * 0.5f, ty1 = tcy - th * 0.5f;
    float tx2 = tcx + tw * 0.5f, ty2 = tcy + th * 0.5f;

    // Intersection
    float ix1 = fmaxf(px1, tx1), iy1 = fmaxf(py1, ty1);
    float ix2 = fminf(px2, tx2), iy2 = fminf(py2, ty2);
    float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);

    // Union
    float area_p = pw * ph;
    float area_t = tw * th;
    float union_area = area_p + area_t - inter;

    float eps = 1e-7f;
    float iou = inter / (union_area + eps);

    // Enclosing box diagonal squared
    float cx1 = fminf(px1, tx1), cy1 = fminf(py1, ty1);
    float cx2 = fmaxf(px2, tx2), cy2 = fmaxf(py2, ty2);
    float c_diag_sq = (cx2 - cx1) * (cx2 - cx1) + (cy2 - cy1) * (cy2 - cy1);

    // Center distance squared
    float d_sq = (pcx - tcx) * (pcx - tcx) + (pcy - tcy) * (pcy - tcy);

    // Aspect ratio consistency
    float atan_pred = atan2f(pw, ph + eps);
    float atan_tgt = atan2f(tw, th + eps);
    float v = (4.0f / (CUDART_PI_F * CUDART_PI_F)) * (atan_pred - atan_tgt) * (atan_pred - atan_tgt);
    float alpha = v / (1.0f - iou + v + eps);

    float ciou = iou - d_sq / (c_diag_sq + eps) - alpha * v;
    losses[i] = 1.0f - ciou;
}


// ============================================================================
// Kernel 3: Fused detection decode
// ============================================================================
// Decodes raw model predictions: apply sigmoid, compute combined scores,
// filter by threshold — all in one pass.

__global__ void fused_detection_decode_kernel(
    const float* __restrict__ predictions,  // (A, 5+C)
    float* __restrict__ boxes,              // (A, 4) cx,cy,w,h
    float* __restrict__ scores,             // (A,) combined score
    int* __restrict__ labels,               // (A,) predicted class
    int A, int C,
    float conf_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A) return;

    const float* pred = predictions + i * (5 + C);

    // Copy box
    boxes[i * 4 + 0] = pred[0];
    boxes[i * 4 + 1] = pred[1];
    boxes[i * 4 + 2] = pred[2];
    boxes[i * 4 + 3] = pred[3];

    // Sigmoid objectness
    float obj = 1.0f / (1.0f + expf(-pred[4]));

    // Find max class score (sigmoid)
    float max_cls = -1e10f;
    int max_idx = 0;
    for (int c = 0; c < C; c++) {
        float val = pred[5 + c];
        if (val > max_cls) {
            max_cls = val;
            max_idx = c;
        }
    }
    float cls_prob = 1.0f / (1.0f + expf(-max_cls));

    float combined = obj * cls_prob;
    scores[i] = (combined >= conf_threshold) ? combined : 0.0f;
    labels[i] = max_idx;
}


// ============================================================================
// Python bindings
// ============================================================================

torch::Tensor fused_multimodal_preprocess(
    torch::Tensor rgb,
    torch::Tensor thermal
) {
    TORCH_CHECK(rgb.device().is_cuda(), "rgb must be on CUDA");
    TORCH_CHECK(thermal.device().is_cuda(), "thermal must be on CUDA");
    TORCH_CHECK(rgb.dtype() == torch::kByte, "rgb must be uint8");
    TORCH_CHECK(thermal.dtype() == torch::kByte, "thermal must be uint8");

    int H = rgb.size(0), W = rgb.size(1);
    auto output = torch::empty({6, H, W}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(rgb.device()));

    int total = H * W * 3;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_multimodal_preprocess_kernel<<<blocks, threads>>>(
        rgb.data_ptr<unsigned char>(),
        thermal.data_ptr<unsigned char>(),
        output.data_ptr<float>(),
        H, W
    );

    return output;
}


torch::Tensor fused_batch_multimodal_preprocess(
    torch::Tensor rgb,
    torch::Tensor thermal
) {
    TORCH_CHECK(rgb.device().is_cuda(), "rgb must be on CUDA");
    int B = rgb.size(0), H = rgb.size(1), W = rgb.size(2);
    auto output = torch::empty({B, 6, H, W}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(rgb.device()));

    int total = B * H * W * 3;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_batch_multimodal_preprocess_kernel<<<blocks, threads>>>(
        rgb.data_ptr<unsigned char>(),
        thermal.data_ptr<unsigned char>(),
        output.data_ptr<float>(),
        B, H, W
    );

    return output;
}


torch::Tensor fused_ciou_loss(
    torch::Tensor pred,
    torch::Tensor target
) {
    TORCH_CHECK(pred.device().is_cuda(), "pred must be on CUDA");
    int N = pred.size(0);
    auto losses = torch::empty({N}, torch::TensorOptions()
        .dtype(torch::kFloat32).device(pred.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_ciou_loss_kernel<<<blocks, threads>>>(
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        losses.data_ptr<float>(),
        N
    );

    return losses.mean();
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_detection_decode(
    torch::Tensor predictions,
    float conf_threshold
) {
    TORCH_CHECK(predictions.device().is_cuda(), "predictions must be on CUDA");
    int A = predictions.size(0);
    int C = predictions.size(1) - 5;

    auto boxes = torch::empty({A, 4}, predictions.options());
    auto scores = torch::empty({A}, predictions.options());
    auto labels = torch::empty({A}, torch::TensorOptions()
        .dtype(torch::kInt32).device(predictions.device()));

    int threads = 256;
    int blocks = (A + threads - 1) / threads;

    fused_detection_decode_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        boxes.data_ptr<float>(),
        scores.data_ptr<float>(),
        labels.data_ptr<int>(),
        A, C, conf_threshold
    );

    return {boxes, scores, labels};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_multimodal_preprocess", &fused_multimodal_preprocess,
          "Fused RGB+Thermal preprocessing (CUDA)");
    m.def("fused_batch_multimodal_preprocess", &fused_batch_multimodal_preprocess,
          "Batch fused RGB+Thermal preprocessing (CUDA)");
    m.def("fused_ciou_loss", &fused_ciou_loss,
          "Fused CIoU loss computation (CUDA)");
    m.def("fused_detection_decode", &fused_detection_decode,
          "Fused detection decode with sigmoid+filter (CUDA)");
}
