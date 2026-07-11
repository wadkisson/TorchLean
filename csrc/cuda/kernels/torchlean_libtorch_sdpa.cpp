// LibTorch SDPA forward/backward bridge (g++; nvcc cannot parse torch headers).
#include <lean/lean.h>
#include <torch/torch.h>
#include "torchlean_cuda_buffer.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

static lean_obj_res ioError(const std::string& message) {
  return lean_io_result_mk_error(
      lean_mk_io_error_other_error(1, lean_mk_string(message.c_str())));
}

static bool checkedMul(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *out = a * b;
  return true;
}

static size_t checkedElements(uint32_t batch, uint32_t n, uint32_t d, const char* what) {
  size_t bn = 0;
  size_t total = 0;
  if (!checkedMul((size_t)batch, (size_t)n, &bn) || !checkedMul(bn, (size_t)d, &total)) {
    throw std::runtime_error(std::string("LibTorch SDPA: ") + what + " size overflow");
  }
  return total;
}

static void requireSize(b_lean_obj_arg object, size_t expected, const char* name) {
  const size_t actual = torchlean_cuda_buffer_unbox(object)->size;
  if (actual != expected) {
    throw std::runtime_error(
        std::string("LibTorch SDPA: ") + name + " buffer size mismatch (expected " +
        std::to_string(expected) + ", got " + std::to_string(actual) + ")");
  }
}

static void validateInputs(b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V,
                           b_lean_obj_arg M, uint32_t hasMask, uint32_t batch,
                           uint32_t n, uint32_t d, b_lean_obj_arg DOut = nullptr) {
  if (hasMask > 1) throw std::runtime_error("LibTorch SDPA: hasMask must be 0 or 1");
  const size_t qkvElements = checkedElements(batch, n, d, "Q/K/V");
  requireSize(Q, qkvElements, "Q");
  requireSize(K, qkvElements, "K");
  requireSize(V, qkvElements, "V");
  if (DOut != nullptr) requireSize(DOut, qkvElements, "dOut");
  if (hasMask != 0) {
    const size_t maskElements = checkedElements(batch, n, n, "mask");
    requireSize(M, maskElements, "mask");
  }
}

static at::Tensor view3(b_lean_obj_arg o, uint32_t batch, int64_t n, int64_t d) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  return at::from_blob(torchlean_cuda_buffer_unbox(o)->data,
                       {static_cast<int64_t>(batch), n, d}, options);
}

static c10::optional<at::Tensor> attnMask(b_lean_obj_arg M, uint32_t hasMask, uint32_t batch,
                                        uint32_t n) {
  if (!hasMask) return c10::nullopt;
  // PyTorch SDPA boolean masks use true for entries that participate in attention.
  return c10::optional<at::Tensor>(view3(M, batch, n, n).to(at::kBool));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_fwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  try {
    validateInputs(Q, K, V, M, hasMask, batch, n, d);
    auto mask = attnMask(M, hasMask, batch, n);
    auto y = at::scaled_dot_product_attention(view3(Q, batch, n, d), view3(K, batch, n, d),
                                              view3(V, batch, n, d), mask, 0., false, scale)
                 .contiguous();
    auto* out = torchlean_cuda_buffer_alloc((size_t)y.numel());
    const cudaError_t copy = cudaMemcpy(out->data, y.data_ptr<float>(),
                                       (size_t)y.numel() * sizeof(float),
                                       cudaMemcpyDeviceToDevice);
    if (copy != cudaSuccess) {
      torchlean_cuda_buffer_drop_unboxed(out);
      return ioError(std::string("LibTorch SDPA forward copy failed: ") + cudaGetErrorString(copy));
    }
    return lean_io_result_mk_ok(torchlean_cuda_buffer_box(out));
  } catch (const c10::Error& e) {
    return ioError(std::string("LibTorch SDPA forward failed: ") + e.what_without_backtrace());
  } catch (const std::exception& e) {
    return ioError(std::string("LibTorch SDPA forward failed: ") + e.what());
  } catch (...) {
    return ioError("LibTorch SDPA forward failed with an unknown native exception");
  }
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_bwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  try {
    validateInputs(Q, K, V, M, hasMask, batch, n, d, DOut);
    at::AutoGradMode enable_grad(true);
    auto mask = attnMask(M, hasMask, batch, n);
    auto q = view3(Q, batch, n, d).detach().requires_grad_(true);
    auto k = view3(K, batch, n, d).detach().requires_grad_(true);
    auto v = view3(V, batch, n, d).detach().requires_grad_(true);
    auto grad_out = view3(DOut, batch, n, d);
    auto out = at::scaled_dot_product_attention(q, k, v, mask, 0., false, scale);
    out.backward(grad_out);
    const size_t numel = checkedElements(batch, n, d, "gradient");
    at::Tensor grads[3] = {q.grad(), k.grad(), v.grad()};
    torchlean_cuda_buffer* bufs[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) bufs[i] = torchlean_cuda_buffer_alloc(numel);
    for (int i = 0; i < 3; ++i) {
      auto t = grads[i].contiguous();
      const cudaError_t copy = cudaMemcpy(bufs[i]->data, t.data_ptr<float>(),
                                         numel * sizeof(float), cudaMemcpyDeviceToDevice);
      if (copy != cudaSuccess) {
        for (auto* buffer : bufs) torchlean_cuda_buffer_drop_unboxed(buffer);
        return ioError(std::string("LibTorch SDPA backward copy failed: ") +
                       cudaGetErrorString(copy));
      }
    }
    return lean_io_result_mk_ok(torchlean_cuda_box_three_buffers(bufs[0], bufs[1], bufs[2]));
  } catch (const c10::Error& e) {
    return ioError(std::string("LibTorch SDPA backward failed: ") + e.what_without_backtrace());
  } catch (const std::exception& e) {
    return ioError(std::string("LibTorch SDPA backward failed: ") + e.what());
  } catch (...) {
    return ioError("LibTorch SDPA backward failed with an unknown native exception");
  }
}
