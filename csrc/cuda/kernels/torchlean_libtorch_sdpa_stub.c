// Optional LibTorch SDPA symbols for native CUDA builds that do not link LibTorch.
#include <lean/lean.h>

static lean_obj_res libtorch_unavailable(const char *operation) {
  lean_object *message = lean_mk_string(operation);
  return lean_io_result_mk_error(lean_mk_io_error_other_error(1, message));
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_fwd(
    b_lean_obj_arg q, b_lean_obj_arg k, b_lean_obj_arg v, b_lean_obj_arg mask,
    uint32_t has_mask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  (void)q;
  (void)k;
  (void)v;
  (void)mask;
  (void)has_mask;
  (void)batch;
  (void)n;
  (void)d;
  (void)scale;
  return libtorch_unavailable(
      "LibTorch SDPA forward is unavailable; rebuild with -K cuda=true -K libtorch=true");
}

LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_bwd(
    b_lean_obj_arg q, b_lean_obj_arg k, b_lean_obj_arg v, b_lean_obj_arg mask,
    b_lean_obj_arg d_out, uint32_t has_mask, uint32_t batch, uint32_t n,
    uint32_t d, double scale) {
  (void)q;
  (void)k;
  (void)v;
  (void)mask;
  (void)d_out;
  (void)has_mask;
  (void)batch;
  (void)n;
  (void)d;
  (void)scale;
  return libtorch_unavailable(
      "LibTorch SDPA backward is unavailable; rebuild with -K cuda=true -K libtorch=true");
}
