#pragma once

#include <lean/lean.h>

#include "torchlean_size_common.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Shared CUDA runtime bits.
// Wrappers validate shapes before launch; `checkCuda` is only for CUDA driver/runtime failures.

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    char detail[512];
    snprintf(detail, sizeof(detail), "%s: %s (%d)", msg, cudaGetErrorString(e), (int)e);
    lean_internal_panic(detail);
  }
}

// CUDA records the last asynchronous launch error per host thread. Clear it immediately before a
// launch when the next check is meant to diagnose that launch, not an older unchecked kernel.
static inline void torchlean_cuda_clear_pending_error() {
  (void)cudaGetLastError();
}

// Check the launch error recorded by the kernel that was just submitted.
static inline void torchlean_cuda_check_launch(const char* msg) {
  checkCuda(cudaGetLastError(), msg);
}

static inline void torchlean_cuda_free_checked(void** ptr, const char* msg) {
  if (ptr && *ptr) {
    checkCuda(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

static inline void torchlean_cuda_lock(pthread_mutex_t* mutex, const char* msg) {
  if (pthread_mutex_lock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_unlock(pthread_mutex_t* mutex, const char* msg) {
  if (pthread_mutex_unlock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

// Stream-ordered scratch freelist (per translation unit that includes this header).
//
// Same rationale as the Buffer pool: eager TorchLean uses the default stream, so reuse after free
// is ordered without CUDA events. Exact-size matching only — callers pass the request size on free,
// so we cannot safely oversize-bucket without a side table.
struct torchlean_cuda_scratch_block {
  size_t bytes;
  void* ptr;
};

static pthread_mutex_t g_torchlean_cuda_scratch_mutex = PTHREAD_MUTEX_INITIALIZER;
static torchlean_cuda_scratch_block* g_torchlean_cuda_scratch_cache = nullptr;
static size_t g_torchlean_cuda_scratch_count = 0;
static size_t g_torchlean_cuda_scratch_cap = 0;
static size_t g_torchlean_cuda_scratch_bytes = 0;

static constexpr size_t kTorchleanCudaScratchMaxBlocks = 1024;
static constexpr size_t kTorchleanCudaScratchMaxBytes = 1ull << 30;  // 1 GiB per TU

static inline void torchlean_cuda_scratch_push(torchlean_cuda_scratch_block block) {
  if (g_torchlean_cuda_scratch_count == g_torchlean_cuda_scratch_cap) {
    size_t new_cap = g_torchlean_cuda_scratch_cap == 0 ? 16 : g_torchlean_cuda_scratch_cap * 2;
    if (new_cap < g_torchlean_cuda_scratch_cap) {
      lean_internal_panic("torchlean_cuda_scratch_push: capacity overflow");
    }
    const size_t alloc_bytes = checked_bytes_size(
        new_cap, sizeof(torchlean_cuda_scratch_block),
        "torchlean_cuda_scratch_push: cache byte size overflow");
    void* next = realloc(g_torchlean_cuda_scratch_cache, alloc_bytes);
    if (!next) {
      lean_internal_panic_out_of_memory();
    }
    g_torchlean_cuda_scratch_cache = (torchlean_cuda_scratch_block*)next;
    g_torchlean_cuda_scratch_cap = new_cap;
  }
  g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count++] = block;
  g_torchlean_cuda_scratch_bytes += block.bytes;
}

static inline void torchlean_cuda_scratch_flush(void) {
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch flush failed");
  torchlean_cuda_scratch_block* blocks = g_torchlean_cuda_scratch_cache;
  size_t count = g_torchlean_cuda_scratch_count;
  g_torchlean_cuda_scratch_cache = nullptr;
  g_torchlean_cuda_scratch_count = 0;
  g_torchlean_cuda_scratch_cap = 0;
  g_torchlean_cuda_scratch_bytes = 0;
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch flush failed");

  for (size_t i = 0; i < count; ++i) {
    torchlean_cuda_free_checked(&blocks[i].ptr, "cudaFree cached scratch block failed");
  }
  free(blocks);
}

static inline void* torchlean_cuda_scratch_alloc_bytes(size_t bytes, const char* msg) {
  if (bytes == 0) {
    return nullptr;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch alloc failed");
  for (size_t i = g_torchlean_cuda_scratch_count; i > 0; --i) {
    const size_t idx = i - 1;
    if (g_torchlean_cuda_scratch_cache[idx].bytes != bytes) {
      continue;
    }
    void* ptr = g_torchlean_cuda_scratch_cache[idx].ptr;
    g_torchlean_cuda_scratch_bytes -= g_torchlean_cuda_scratch_cache[idx].bytes;
    g_torchlean_cuda_scratch_cache[idx] =
        g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count - 1];
    g_torchlean_cuda_scratch_count--;
    torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                          "pthread_mutex_unlock scratch alloc failed");
    return ptr;
  }
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch alloc failed");

  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    torchlean_cuda_scratch_flush();
    err = cudaMalloc(&ptr, bytes);
  }
  checkCuda(err, msg);
  return ptr;
}

static inline void torchlean_cuda_scratch_free_bytes(void** ptr, size_t bytes, const char* msg) {
  (void)msg;
  if (!ptr || !*ptr) {
    return;
  }
  if (bytes == 0) {
    torchlean_cuda_free_checked(ptr, msg);
    return;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_lock scratch free failed");
  const bool over_cap = g_torchlean_cuda_scratch_count >= kTorchleanCudaScratchMaxBlocks ||
                        g_torchlean_cuda_scratch_bytes + bytes > kTorchleanCudaScratchMaxBytes;
  if (over_cap) {
    torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                          "pthread_mutex_unlock scratch free failed");
    torchlean_cuda_free_checked(ptr, "cudaFree scratch over cache cap failed");
    return;
  }
  torchlean_cuda_scratch_push({bytes, *ptr});
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "pthread_mutex_unlock scratch free failed");
  *ptr = nullptr;
}

template <typename T>
static inline T* torchlean_cuda_scratch_alloc(size_t count, const char* msg) {
  return (T*)torchlean_cuda_scratch_alloc_bytes(checked_bytes_size(count, sizeof(T), msg), msg);
}

template <typename T>
static inline void torchlean_cuda_scratch_free(T** ptr, size_t count, const char* msg) {
  torchlean_cuda_scratch_free_bytes((void**)ptr, checked_bytes_size(count, sizeof(T), msg), msg);
}
