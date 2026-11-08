# Revision after independent critical review — 8 November 2026

This protocol is written before the additional numerical runs. It supplements,
and does not replace, evidence snapshot `db39238` and manuscript `7f2a9d8`.
The review is post hoc. New experiments below are explicitly revision analyses,
not retroactive preregistration of the original campaign.

## Questions and limits

1. Audit every quantitative criticism against the original summaries. Replay
   projected updates on all 432 original streams and count every changed state,
   not only equality of rounded aggregate endpoints.
2. Expose ideal-input error, noisy-input error, observed maximum error, and an
   oracle inverse noise benchmark. The oracle uses hidden generator parameters
   only as an evaluator. Its error is not a universal lower bound or a GUM
   uncertainty estimate. MAE components are not subtracted as if additive.
3. Check whether fivefold cross-validation changes the capacity result. Compare
   the existing held-out selector and fivefold CV with the same lambda grid and
   float32 storage. Both spend the same B references, then refit all B.
4. Check sensitivity to declared generator assumptions, without interpreting
   simulated-unit intervals as evidence about manufactured sensor populations.
5. Compare fixed and decaying NLMS with simple affine recalibration and causal
   batch refits; extend RLS forgetting-factor selection. Initial models and
   transforms remain identical across update methods.

## Static revision matrix

New seed 2026110811; eight virtual units and two repetitions; B = 30/75.
Nominal study: LUT float32 and compact cubic at 80/144/224 B, each with original
held-out selection and deterministic interleaved fivefold CV. Include the
selected Chebyshev polynomial with both selection procedures and fitted physics.
The lambda grid remains {0, 1e-6, 1e-4, 1e-2, 1, 100}; polynomial degrees remain
{1, 2, 3, 4, 5, 7, 9}. CV uses each reference for validation once; no test labels.

One-factor-at-a-time sensitivity: nominal; per-conversion noise 0.25 or 4 LSB
(nominal 1); tolerance half-width multiplier 0.5 or 2 (nominal 1); distorted
NTC sinusoid amplitude 1 or 12 LSB (nominal 4); distorted NTC sinusoid frequency
1 or 9 cycles over normalized ADC (nominal 3). Amplitude/frequency changes apply
only to distorted NTC. No factorial combination or post-result setting removal.
Use the 144-byte LUT/cubic CV, polynomial CV and physics. The same unit/noise
seeds are paired across settings. Label noise and 16-conversion averaging stay
unchanged. Store all fitted models, references, ideal/noisy predictions and
oracle errors. Report scenario-specific metrics, not mixed physical units.

The C2 Horner representation remains an efficient implementation of a smooth
source spline with redundant expanded storage. It will no longer support the
headline claim that memory changes which method is best. Independent free
piece coefficients would change the continuity/function-class assumptions;
no general claim against all piecewise cubics is made. The primary comparison
includes compact cubics and smaller physical/polynomial references.

## Sequential revision matrix

Development seed 2026110810, four units/two repetitions. Confirmation seed
2026110811, eight units/two repetitions. All three scenarios, both budgets and
all three original drift schedules. Use original initial fits and transforms.
The reference stream remains 384 labels, in shuffled full-range 32-bin blocks.

Retain the original nine methods. Additional methods:

- LUT/cubic decaying NLMS: eta(t)=0.5/(1+t/tau), t starts at 1; tau selected
  globally per family from {32,128,512} on development only.
- Cubic/polynomial RLS: extend the global forgetting-factor grid to
  {0.90,0.95,0.98,0.995,1.0}; retain original 0.98 as a separate reference.
- Two-reference affine correction of each frozen LUT/cubic output: every
  completed block of 32 new references, select the lowest/highest frozen
  prediction and fit offset/gain from those two labelled observations. Apply
  only after that block, never before its labels arrive. Previous correction
  stays active during the next block. This is a causal two-point baseline.
- LUT/cubic batch refit using all initial and prefix references, and using
  the latest 64 references from that combined history. Refit at the declared
  checkpoints using the initial lambda without reselection. Predict before
  the current update for pre-update diagnostics. Account for retained data;
  these are larger-memory, untimed accuracy references, not MCU equivalents.

Development selection uses the original checkpoint-integrated normalized MSE
including no drift. Retain all candidates and numerical failures. No settings
are selected using confirmation errors. Report final and integrated errors,
including no-drift losses; count activity of projected variants.

## Hardware follow-up and interpretation

Prepare a separate acquisition suite for the same predetermined synthetic
fixture, adding the fitted Chebyshev polynomial and physical model to the four
measured representations. Preserve original captures and firmware. Verify
target float32 arithmetic before asking for new measurements. Add an ESP32-S3
environment as an optional FPU target, contingent on the actual available board.
Host tests and cross-compilation do not count as hardware timing or energy.

Existing ordinary NLMS timings, rather than inactive full-PAVA timings, become
the principal update comparison. Explain that pre-expanding all 29 cubic32
pieces requires 480 B including transforms; it cannot be called a 144 B
implementation. Show the local unrolled kernel and avoid claiming optimality.

The end-to-end ADC example is an analytical budget, not a measured acquisition
cycle: 16*13*(128/16 MHz)=1664 us for steady-state conversions. Include kernel
time and state all omissions. Energy, WCET, peak stack and between-board
variation remain unmeasured unless separately acquired.

Physical sensor measurements cannot be supplied by this host revision. Provide
a small prospective NTC/reference campaign plan with uncertainty components,
without fabricated readings or a claim of mandatory journal policy. A public
raw-data DOI is not created unless a deposit can actually be completed.
