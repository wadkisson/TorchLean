// Stubs when TorchLean is built without `-K libtorch=true`.
#include <lean/lean.h>
#include "torchlean_cuda_buffer.h"

static lean_obj_res ioError(const char* message) {
  return lean_io_result_mk_error(
      lean_mk_io_error_other_error(1, lean_mk_string(message)));
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_fwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  (void)Q; (void)K; (void)V; (void)M; (void)hasMask; (void)batch; (void)n; (void)d; (void)scale;
  return ioError(
      "LibTorch SDPA forward unavailable; rebuild with `-K cuda=true -K libtorch=true`");
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_bwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  (void)Q; (void)K; (void)V; (void)M; (void)DOut; (void)hasMask; (void)batch; (void)n; (void)d;
  (void)scale;
  return ioError(
      "LibTorch SDPA backward unavailable; rebuild with `-K cuda=true -K libtorch=true`");
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_fwd_buf(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  (void)Q; (void)K; (void)V; (void)M; (void)hasMask; (void)batch; (void)n; (void)d; (void)scale;
  lean_internal_panic(
      "LibTorch SDPA forward unavailable; rebuild with `-K cuda=true -K libtorch=true`");
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_bwd_buf(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  (void)Q; (void)K; (void)V; (void)M; (void)DOut; (void)hasMask; (void)batch; (void)n; (void)d;
  (void)scale;
  lean_internal_panic(
      "LibTorch SDPA backward unavailable; rebuild with `-K cuda=true -K libtorch=true`");
}
