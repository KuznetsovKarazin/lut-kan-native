# Prospective physical validation

No physical NTC/LDR calibration campaign is included in the current article.
A useful next step is several NTC units against a traceably calibrated reference,
with calibration and held-out setpoints fixed before model fitting. Repeat
heating/cooling traversals and whole acquisition cycles. Retain time-resolved ADC
codes, reference readings, unit IDs, supply/reference settings, excitation,
settling criteria and environmental stability.

A preliminary dry setup can use 10 kohm NTCs (for example Vishay
NTCLE100E3103JB0), precision divider resistors, and a TMP117 comparison module.
A low-cost comparison sensor is not a calibrated reference for the assembled
system. Component cost alone does not provide traceability or an uncertainty
budget. Obtain a reference instrument or calibration certificate through the
laboratory before making absolute-accuracy claims.

Use the divider orientation assumed by the code: supply -> fixed resistor ->
ADC node -> NTC -> ground. Record whether divider supply and ADC reference are
ratiometric. Keep sensing elements close in a controlled dry environment, with
MCUs and dissipating resistors outside the temperature region. Do not immerse
bare components or unsealed breakout boards in water; select sealed probes if
a bath is required and follow their specifications.

Measure self-heating by comparing continuous and switched excitation. For two
10 kohm resistors at 5 V, the NTC dissipates about 0.625 mW. This calculation does
not determine its actual temperature rise. Use the selected part's R/T data;
the simulator's nominal beta is not a substitute for a component specification.

The uncertainty budget should include reference calibration/drift, bath or
chamber stability and gradients, excitation/self-heating, thermal lag/hysteresis,
leads/contacts, ADC nonlinearity and quantization, reference-voltage drift and
repeatability. Regression MAE is reported separately.

Manufacturer references:
- https://www.vishay.com/docs/29049/ntcle100.pdf
- https://www.ti.com/lit/ds/symlink/tmp117.pdf
- https://www.adafruit.com/product/4821

This is a prospective method outline, not a shopping quote or a completed
metrological campaign. The current implementation study uses controlled synthetic
accuracy cases and measured execution on development boards.
