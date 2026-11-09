# Second-review refinement protocol — 8 November 2026

This addendum is written before the following calculations. It reuses the fixed
second-stage development seed 2026110810 and evaluation seed 2026110811, with the
same units (4 development / 8 evaluation), two repetitions, three scenarios,
two initial budgets and three drift schedules. Evaluation is a paired post-review
reanalysis of already inspected streams, not a new untouched test set.

1. Replace the weak two-endpoint affine correction in the main comparison with
   least-squares affine correction of each frozen model's normalized output over
   every completed block of 32 labelled references. Accumulate sum(p), sum(y),
   sum(p*p), sum(p*y); solve gain/intercept after the complete block, reset sums.
   Use all labels; never use a partial/future block. A degenerate block retains
   the previous correction. The four float32 sums cost 16 B, plus 8 B for the
   current gain/intercept and an integer block counter, in addition to the base
   model. Maintain an independent float64 calculation to assess roundoff.
2. Extend decaying-NLMS tau candidates from {32,128,512} to
   {32,128,512,1024,2048}. Use the unchanged integrated normalized-MSE development
   objective across every scenario/budget/drift. Report a boundary optimum if it
   remains at a boundary; ordinary NLMS provides the constant-step reference.
3. Retain original and new methods in the archived results. Recompute only the
   added algorithms and compare unchanged original methods from the saved
   revision summaries. No selective removal of unfavorable scenarios.
4. Prepare a seven-implementation MCU suite on the single matched fixture:
   four previous 144-byte representations, fitted physical model, and the same
   selected degree-9 global polynomial in Chebyshev/Clenshaw and converted
   power-basis/Horner forms (both 56 B). Coefficient conversion is offline;
   float32 stored-coefficient and target-kernel errors must be checked, including
   the declared test range. Clenshaw/Horner timing is not assumed identical.
   New firmware identity/protocol versions prevent accepting old six-model logs.
   Acquire at least one full session per board; two default warm sessions remain
   inexpensive and expose session agreement. No physical MCU results will be
   synthesized in the host environment. Include a strict common-fixture merger.
5. Fix plot annotations and explicit logarithmic tick labels; migrate to the
   publisher-provided Sensors Journal template and document its provenance.
   A public data DOI is a separate deposit step; no DOI will be invented.
