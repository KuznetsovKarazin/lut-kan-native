# Evidence recovered during revision

After freezing REVIEW_REVISION_PROTOCOL.md and running the additional numerical
stages, inspection of the earlier optimized-suite archives recovered real timings
for the fitted physical model and degree-9 Chebyshev polynomial on both boards.
The first manuscript had omitted these measurements. They use seed 20261108,
not the matched campaign's 2026110801, and must be presented as a separate panel
with their own inputs, payloads and accuracy. Their four raw logs were reparsed
and compared exactly with saved JSON and sidecar hashes. No new MCU measurement
was performed. See recover_reference_timings.py and recovered_hardware.json.

Consequently, another run on the same two boards is useful for consolidating all
six models into one matched fixture, but is not necessary merely to establish
that these reference implementations were physically timed. The prepared suite
and optional ESP32-S3 environment remain prospective; host validation and builds
must not be presented as real board results.

A public data DOI remains pending. Do not describe a prepared deposit or release
metadata file as a published archive. The manuscript is a working revision and
has not been submitted.
