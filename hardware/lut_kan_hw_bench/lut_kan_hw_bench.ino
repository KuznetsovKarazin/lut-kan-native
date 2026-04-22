/*
 * lut_kan_hw_bench.ino
 * Hardware benchmark for lut-kan-native project
 *
 * Tests K=4 with L in {8, 16, 32} - the three configs of interest
 * Works on: Arduino Mega 2560 (AVR, 8 KB SRAM) and ESP32-C3 (RISC-V, 400 KB)
 *
 * What this measures:
 *   1. On-device training: SGD + lambda2, zero init, best-epoch selection
 *   2. Test MSE after training vs hardcoded post-train-LUT baseline
 *   3. Inference latency: LUT (gather+lerp) vs polynomial (Horner, deg 16)
 *
 * Expected output (saturating target: tanh(4x)+0.15x):
 *   K=4,L= 8   ratio ~  25x   (sim baseline: 23x at n=500)
 *   K=4,L=16   ratio ~ 800x   (sim baseline: 936x at n=500)
 *   K=4,L=32   ratio ~20000x  (sim baseline: 21012x at n=500)
 *
 * Timing (rough expected):
 *   Arduino Mega (16 MHz, soft-float):  LUT ~40 uss/call, Poly ~120 uss/call
 *   ESP32-C3 (160 MHz, soft-float):     LUT ~1-2 uss/call, Poly ~5-8 uss/call
 *
 * Memory (largest config K=4,L=32, float32):
 *   lut + grad + best_lut = 3 x 128 x 4 = 1536 bytes
 *   Well within Arduino Mega's 8 KB SRAM.
 *
 * -- HOW TO USE --------------------------------------------------------------
 * Arduino IDE:
 *   1. Open this file
 *   2. Select board: Tools -> Board -> Arduino Mega 2560  (or ESP32C3 Dev Module)
 *   3. Upload -> open Serial Monitor at 115200 baud
 *
 * PlatformIO: copy to src/main.cpp, configure platformio.ini accordingly.
 * ----------------------------------------------------------------------------
 */

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <Arduino.h>

// --- Configuration ------------------------------------------------------------
#define KK        4       // number of segments (fixed for this sweep)
#define L_MAX     32      // largest L we will test (sets array sizes)
#define N_TRAIN   200     // training points, generated on-the-fly
#define N_VAL     50      // validation points for best-epoch selection
#define N_TEST    200     // test points (deterministic grid)
#define EPOCHS    800     // training epochs
#define LAMBDA2   1.0f    // second-diff regularization weight

// Post-train LUT baseline MSE (from Python simulation, tanh(4x)+0.15x target)
// These are the MSE you'd get by fitting a degree-16 Chebyshev poly and sampling it into the LUT.
// Direct-LUT beats these - the ratio is the paper's main metric.
static const float POST_LUT_MSE[3] = {3.7201e-3f, 9.7252e-4f, 2.4582e-4f};
static const int   L_VALUES[3]     = {8, 16, 32};

// Reciprocal of segment width: replaces division with multiply in inference.
// For K=4, X in [-1,1]: seg_w=0.5, INV_SEG_W=2.0 (exact). ~3x cheaper on soft-float.
static const float INV_SEG_W = (float)KK / 2.0f;

// --- Global arrays (declared at file scope to avoid stack overflow on AVR) ---
static float lut[KK * L_MAX];
static float grad_buf[KK * L_MAX];
static float best_lut[KK * L_MAX];

// Chebyshev-fitted degree-16 polynomial for tanh(4x)+0.15x (power basis, precomputed in Python)
// Used for inference timing comparison only - coefficients degree 0..16
static const float POLY_COEFFS[17] = {
     0.00000000f,   // x^0
     4.10176487f,   // x^1
     0.00000000f,   // x^2
   -18.34776715f,   // x^3
     0.00000000f,   // x^4
    75.75041994f,   // x^5
     0.00000000f,   // x^6
  -209.10284904f,   // x^7
     0.00000000f,   // x^8
   361.41224250f,   // x^9
     0.00000000f,   // x^10
  -372.48039076f,   // x^11
     0.00000000f,   // x^12
   208.41844313f,   // x^13
     0.00000000f,   // x^14
   -48.60347524f,   // x^15
     0.00000000f    // x^16
};

// --- Platform-independent microsecond timer ----------------------------------
static inline uint32_t get_us() {
  return (uint32_t)micros();
}

// --- Free SRAM estimate ------------------------------------------------------
#if defined(ARDUINO_AVR_MEGA2560) || defined(__AVR__)
static int free_sram() {
  extern int __heap_start, *__brkval;
  int v;
  return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval);
}
#else
// ESP32: use heap info
#include "esp_heap_caps.h"
static int free_sram() {
  return (int)heap_caps_get_free_size(MALLOC_CAP_8BIT);
}
#endif

// --- Target function ---------------------------------------------------------
// tanh(4x) + 0.15x  - the "saturating" target from the paper
static inline float target_fn(float x) {
  return tanhf(4.0f * x) + 0.15f * x;
}

// --- LUT inference -----------------------------------------------------------
// Faithful port of core.py LUTEdge.forward():
//   half-open segment sampling, endpoint-inclusive interpolation indices.
static float lut_infer(float x, const float* lut_ptr, int L) {
  if (x < -1.0f) x = -1.0f;
  if (x >= 1.0f)  x = 1.0f - 1e-6f;

  // Multiply by reciprocal instead of dividing (3-5x cheaper on soft-float MCU)
  float t = (x + 1.0f) * INV_SEG_W;
  int   k = (int)t;
  if (k < 0) k = 0;
  if (k >= KK) k = KK - 1;

  float u = t - (float)k;
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;

  float pos = u * (float)(L - 1);
  int   r0  = (int)pos;
  if (r0 < 0) r0 = 0;
  if (r0 >= L - 1) r0 = L - 2;
  int   r1  = r0 + 1;
  float w   = pos - (float)r0;

  const float* seg = lut_ptr + (k * L);
  return seg[r0] * (1.0f - w) + seg[r1] * w;
}

// --- Polynomial inference (Horner, for timing comparison) --------------------
static float poly_infer(float x) {
  float r = POLY_COEFFS[16];
  for (int i = 15; i >= 0; i--) {
    r = r * x + POLY_COEFFS[i];
  }
  return r;
}

// --- LUT backward (accumulate gradient for one sample) -----------------------
// Computes dL/d_lut for MSE loss: adds 2*(pred - y) * interpolation_weights
static void lut_backward_sample(float x, float residual, float* grad_ptr, int L) {
  if (x < -1.0f) x = -1.0f;
  if (x >= 1.0f)  x = 1.0f - 1e-6f;

  float t = (x + 1.0f) * INV_SEG_W;
  int   k = (int)t;
  if (k < 0) k = 0;
  if (k >= KK) k = KK - 1;

  float u = t - (float)k;
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;

  float pos = u * (float)(L - 1);
  int   r0  = (int)pos;
  if (r0 < 0) r0 = 0;
  if (r0 >= L - 1) r0 = L - 2;
  int   r1  = r0 + 1;
  float w   = pos - (float)r0;

  // MSE gradient: 2 * residual * weight (sparse: only 2 cells per sample)
  float g = 2.0f * residual;
  grad_ptr[k * L + r0] += g * (1.0f - w);
  grad_ptr[k * L + r1] += g * w;
}

// --- lambda2 second-difference regularization gradient ----------------------------
// Ports regularizers.py::second_diff_penalty gradient exactly.
// Loss = lambda2 * mean_{k,i} (lut[k,i+2] - 2*lut[k,i+1] + lut[k,i])^2
// Gradient at lut[k,j] = 2*lambda2/n_terms * (d2[j-2] - 2*d2[j-1] + d2[j])
// where d2[i] = lut[k,i] - 2*lut[k,i+1] + lut[k,i+2], d2[i]=0 outside [0,L-3]
static void add_lambda2_grad(int L) {
  if (L < 3) return;
  const int n_terms = KK * (L - 2);  // number of second-difference terms
  const float scale = 2.0f * LAMBDA2 / (float)n_terms;

  for (int k = 0; k < KK; k++) {
    float* seg  = lut      + k * L;
    float* gseg = grad_buf + k * L;

    // Compute d2 for this segment (stack allocated, small: L-2 floats)
    // Max L=32 -> 30 floats = 120 bytes. Safe on both AVR and ESP32.
    float d2[L_MAX - 2];
    for (int i = 0; i <= L - 3; i++) {
      d2[i] = seg[i] - 2.0f * seg[i + 1] + seg[i + 2];
    }

    // Accumulate gradient per cell
    for (int j = 0; j < L; j++) {
      float g = 0.0f;
      if (j >= 2)             g += d2[j - 2];  // from term centered at i=j-2
      if (j >= 1 && j <= L-2) g -= 2.0f * d2[j - 1];  // from term at i=j-1
      if (j <= L - 3)         g += d2[j];      // from term at i=j
      gseg[j] += scale * g;
    }
  }
}

// --- Compute MSE on a deterministic test grid ---------------------------------
static float compute_test_mse(int L) {
  float mse = 0.0f;
  for (int i = 0; i < N_TEST; i++) {
    float x = -1.0f + 2.0f * (float)i / (float)(N_TEST - 1);
    float y = target_fn(x);
    float p = lut_infer(x, lut, L);
    float d = p - y;
    mse += d * d;
  }
  return mse / (float)N_TEST;
}

static float compute_val_mse(int L) {
  float mse = 0.0f;
  for (int i = 0; i < N_VAL; i++) {
    // Validation grid: offset by half-step from training grid
    float x = -0.995f + 2.0f * (float)i / (float)(N_VAL - 1);
    float y = target_fn(x);
    float p = lut_infer(x, lut, L);
    float d = p - y;
    mse += d * d;
  }
  return mse / (float)N_VAL;
}

// --- SGD training loop for one L configuration -------------------------------
// lr is L-dependent (empirically determined from simulation):
//   L=8 -> 0.5,  L=16 -> 1.0,  L=32 -> 2.0   (scales as L/16)
static float train_one(int L, float lr, int* best_ep_out) {
  int   total_cells = KK * L;
  float best_val = 1e20f;
  int   best_ep  = 0;

  memset(lut, 0, total_cells * sizeof(float));
  memset(best_lut, 0, total_cells * sizeof(float));

  for (int ep = 0; ep < EPOCHS; ep++) {
    // Zero gradient buffer
    memset(grad_buf, 0, total_cells * sizeof(float));

    // Accumulate data gradients (full-batch, data computed on-the-fly)
    for (int i = 0; i < N_TRAIN; i++) {
      float x    = -1.0f + 2.0f * (float)i / (float)(N_TRAIN - 1);
      float y    = target_fn(x);
      float pred = lut_infer(x, lut, L);
      float res  = pred - y;
      lut_backward_sample(x, res, grad_buf, L);
    }

    // Normalise data gradient by N_TRAIN (matches Python's .mean() loss)
    float inv_n = 1.0f / (float)N_TRAIN;
    for (int j = 0; j < total_cells; j++) grad_buf[j] *= inv_n;

    // Add lambda2 regularization gradient
    add_lambda2_grad(L);

    // SGD step
    for (int j = 0; j < total_cells; j++) {
      lut[j] -= lr * grad_buf[j];
    }

    // Validation MSE -> best-epoch selection
    float val_mse = compute_val_mse(L);
    if (val_mse < best_val) {
      best_val = val_mse;
      best_ep  = ep;
      memcpy(best_lut, lut, total_cells * sizeof(float));
    }

    // Progress every 100 epochs
    if (ep % 100 == 0) {
      Serial.print("    ep "); Serial.print(ep);
      Serial.print("  val_mse="); Serial.print(val_mse, 7);
      Serial.print("  test_mse="); Serial.println(compute_test_mse(L), 7);
    }
  }

  // Restore best weights
  memcpy(lut, best_lut, total_cells * sizeof(float));
  if (best_ep_out) *best_ep_out = best_ep;
  return compute_test_mse(L);
}

// --- setup() - runs once -----------------------------------------------------
void setup() {
  Serial.begin(115200);
  // Wait for serial monitor to connect (especially important on ESP32-C3 USB CDC).
  // Without this delay the latency section runs before the monitor is open.
  unsigned long _t0 = millis();
  while (!Serial && millis() - _t0 < 5000) { delay(10); }
  delay(1500);

  Serial.println(F(""));
  Serial.println(F("========================================"));
  Serial.println(F("  lut-kan-native hardware benchmark"));
  Serial.println(F("  target: tanh(4x) + 0.15x (saturating)"));
  Serial.println(F("========================================"));
  Serial.print(F("Platform: "));
#if defined(ARDUINO_AVR_MEGA2560)
  Serial.println(F("Arduino Mega 2560 (AVR 16 MHz, 8 KB SRAM)"));
#elif defined(CONFIG_IDF_TARGET_ESP32C3) || defined(ARDUINO_ESP32C3_DEV)
  Serial.println(F("ESP32-C3 (RISC-V 160 MHz, 400 KB SRAM)"));
#else
  Serial.println(F("Unknown"));
#endif
  Serial.print(F("Free SRAM at start: ")); Serial.print(free_sram()); Serial.println(F(" bytes"));
  Serial.print(F("K=")); Serial.print(KK);
  Serial.print(F("  N_TRAIN=")); Serial.print(N_TRAIN);
  Serial.print(F("  EPOCHS=")); Serial.print(EPOCHS);
  Serial.print(F("  LAMBDA2=")); Serial.println(LAMBDA2);

  // -- 1. Training sweep: K=4, L in {8, 16, 32} ------------------------------
  Serial.println(F(""));
  Serial.println(F("-- On-device training sweep -----------------------------"));
  Serial.println(F("  (zero init -> SGD + lambda2 -> best-epoch selection)"));

  for (int ci = 0; ci < 3; ci++) {
    int   L        = L_VALUES[ci];
    float lr       = (float)L / 16.0f;   // L=8->0.5, L=16->1.0, L=32->2.0
    int   kxl      = KK * L;
    float density  = (float)N_TRAIN / (float)kxl;
    float post_mse = POST_LUT_MSE[ci];

    Serial.println(F(""));
    Serial.print(F("  > K=4, L=")); Serial.print(L);
    Serial.print(F("  ("));        Serial.print(kxl * 1);      // uint8 bytes for inference
    Serial.print(F(" bytes uint8, "));
    Serial.print(density, 1); Serial.println(F(" pts/cell)"));
    Serial.print(F("    lr=")); Serial.print(lr, 1);
    Serial.print(F("  KxL=")); Serial.print(kxl);
    Serial.print(F("  rule KxL<N? ")); Serial.println(kxl < N_TRAIN ? F("YES YES") : F("NO NO (expect collapse)"));

    uint32_t t_train_start = get_us();
    int  best_ep  = 0;
    float test_mse = train_one(L, lr, &best_ep);
    uint32_t t_train = get_us() - t_train_start;

    float ratio = post_mse / test_mse;

    Serial.println(F("    -- Results ------------------------------------------"));
    Serial.print(F("    Training time: "));
    if (t_train > 1000000UL) {
      Serial.print((float)t_train / 1e6f, 1); Serial.println(F(" s"));
    } else {
      Serial.print(t_train / 1000UL); Serial.println(F(" ms"));
    }
    Serial.print(F("    Best epoch: "));    Serial.println(best_ep);
    Serial.print(F("    Direct-LUT MSE: ")); Serial.println(test_mse, 8);
    Serial.print(F("    Post-LUT   MSE: ")); Serial.println(post_mse, 8);
    Serial.print(F("    Ratio (post/direct): ")); Serial.print(ratio, 1);
    Serial.println(F("x"));
    Serial.print(F("    Sim expected: "));
    if (L == 8)  Serial.println(F("~25x (sim: 23x at n=500)"));
    if (L == 16) Serial.println(F("~800x (sim: 936x at n=500)"));
    if (L == 32) Serial.println(F("~20000x (sim: 21012x at n=500)"));
    Serial.print(F("    Free SRAM now: ")); Serial.print(free_sram()); Serial.println(F(" bytes"));
  }

  // -- 3. Summary ------------------------------------------------------------
  Serial.println(F(""));

  // -- 2. Inference latency benchmark ----------------------------------------
  Serial.println(F(""));
  Serial.println(F("-- Inference latency (1000 calls each) ------------------"));

  // Init LUT with something non-zero for timing (doesn't affect accuracy)
  for (int j = 0; j < KK * L_MAX; j++) lut[j] = 0.01f * (j + 1);

  // Pre-compute x values ONCE so the benchmark measures only infer cost,
  // not x-generation (which contains a division and contaminates both timings equally,
  // compressing the ratio toward 1.0).
  // 200 values * 4 bytes = 800 bytes -- fine on both Mega (6 KB free) and ESP32-C3.
  static float x_bench[N_TEST];
  for (int i = 0; i < N_TEST; i++) {
    x_bench[i] = -1.0f + 2.0f * (float)i / (float)(N_TEST - 1);
  }

  volatile float sink = 0.0f;

  // LUT timing: fast variant (L=32 is power-of-2, log2=5)
  uint32_t t0 = get_us();
  for (int rep = 0; rep < 5; rep++) {
    for (int i = 0; i < N_TEST; i++) {
      sink += lut_infer_fast(x_bench[i], lut, 32, 5);
    }
  }
  uint32_t t_lut = get_us() - t0;

  // Also measure general lut_infer for comparison
  uint32_t t0g = get_us();
  for (int rep = 0; rep < 5; rep++) {
    for (int i = 0; i < N_TEST; i++) {
      sink += lut_infer(x_bench[i], lut, 32);
    }
  }
  uint32_t t_lut_general = get_us() - t0g;

  // Polynomial timing: same loop structure
  t0 = get_us();
  for (int rep = 0; rep < 5; rep++) {
    for (int i = 0; i < N_TEST; i++) {
      sink += poly_infer(x_bench[i]);
    }
  }
  uint32_t t_poly = get_us() - t0;

  (void)sink;

  Serial.print(F("  LUT fast K=4,L=32  1000 calls: ")); Serial.print(t_lut); Serial.println(F(" us"));
  Serial.print(F("  LUT gen  K=4,L=32  1000 calls: ")); Serial.print(t_lut_general); Serial.println(F(" us"));
  Serial.print(F("  Poly deg=16        1000 calls: ")); Serial.print(t_poly); Serial.println(F(" us"));
  Serial.print(F("  Speed ratio (poly/lut_fast): "));
  Serial.print((float)t_poly / (float)t_lut, 2);
  Serial.println(F("x"));
  Serial.print(F("  Speed ratio (poly/lut_general): "));
  Serial.print((float)t_poly / (float)t_lut_general, 2);
  Serial.println(F("x  (general variant used during training)"));
  Serial.println(F("  Note: 6x on x86 FPU; lower here: soft-float float-to-int cost"));


  Serial.println(F("========================================"));
  Serial.println(F("  Done. Copy output to compare with sim."));
  Serial.println(F("  Expected: ratios should match within ~2-3x"));
  Serial.println(F("  of simulation values (different n, optimizer state)"));
  Serial.println(F("========================================"));
}

void loop() {
  // Nothing - all work done in setup()
}

 // ---- Fast LUT inference (L must be a power of 2) -------------------------
// Optimization: combines k and r0 computation into ONE float-to-int cast
// instead of two, using integer bit-ops for the second index.
// Only valid when L is a power of 2: L=8,16,32 all qualify.
//
// Cycles saved on RISC-V soft-float: ~200 cycles per call (one fcvt.w.s removed)
// Expected speedup vs lut_infer: ~1.5-2x for the infer-only cost.
static float lut_infer_fast(float x, const float* lut_ptr, int L, int log2_L) {
  if (x < -1.0f) x = -1.0f;
  if (x >= 1.0f)  x = 1.0f - 1e-7f;

  // Map x in [-1,1) -> fp in [0, K*L) in ONE step
  // scale = K*L/2 = total_cells/2; for K=4,L=32: scale=64
  float scale = (float)(KK * L) * 0.5f;
  float fp    = (x + 1.0f) * scale;      // [0, K*L)

  int idx = (int)fp;                      // ONE float-to-int (was: two)
  int k   = idx >> log2_L;               // segment index  (integer shift)
  int r0  = idx & (L - 1);               // cell within segment (integer AND)
  int r1  = r0 + 1; if (r1 >= L) r1 = L - 1;
  float w = fp - (float)idx;             // fractional weight (int->float, cheap)

  const float* seg = lut_ptr + k * L;
  return seg[r0] * (1.0f - w) + seg[r1] * w;
}

 