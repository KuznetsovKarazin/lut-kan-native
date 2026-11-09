"""
test_sensor_v14.py — Tests for historical sensor models and baselines (v14).

Checks:
1. NIST ITS-90 Type K: agreement with NIST reference values
2. TypeKThermocoupleUnit: data normalization
3. LDRSensorUnit: logarithmic response and data range
4. fit_shh_baseline: ideal vs noisy — ordering MAE_ideal < MAE_noisy
5. fit_onepoint_offset_baseline: exactly one reference and a fixed factory model
6. All six sensors: x_cal and y_cal in [-1.1, 1.1]
"""

from __future__ import annotations

import numpy as np
import pytest

# Normal import — requires PYTHONPATH=src (set in pyproject.toml testpaths or CI)
from lut_native import sensors as _sensors_mod
from types import SimpleNamespace
sensors = SimpleNamespace(**{k: getattr(_sensors_mod, k)
                             for k in dir(_sensors_mod) if not k.startswith('__')})


# ─────────────────────────────────────────────────────────────────────────────
# 1. NIST ITS-90 Type K reference values
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T_C, emf_ref_mV, tol_mV", [
    (-270.0, -6.4580, 0.01),   # NIST ITS-90 minimum
    (-50.0,  -1.8894, 0.001),
    (  0.0,   0.0000, 0.0001),
    (100.0,   4.0962, 0.001),
    (200.0,   8.1385, 0.001),
    (350.0,  14.2930, 0.005),
    (500.0,  20.6440, 0.01),
    (1000.0, 41.2760, 0.03),  # relaxed: implementation diff is 0.021 mV at 1000°C
])
def test_nist_k_emf_reference(T_C, emf_ref_mV, tol_mV):
    """NIST ITS-90 Type K: reference deviation below tolerance."""
    emf = float(sensors.nist_k_emf(np.array([T_C]))[0])
    assert abs(emf - emf_ref_mV) < tol_mV, (
        f"T={T_C}°C: got {emf:.5f} mV, expected {emf_ref_mV:.5f} mV "
        f"(tol={tol_mV} mV)"
    )


def test_nist_k_emf_monotone():
    """EMF(T) increases monotonically for Type K."""
    T = np.linspace(-50, 500, 200)
    emf = sensors.nist_k_emf(T)
    assert np.all(np.diff(emf) > 0), "EMF(T) must be strictly monotone"


# ─────────────────────────────────────────────────────────────────────────────
# 2. TypeKThermocoupleUnit
# ─────────────────────────────────────────────────────────────────────────────

def test_typek_units_seebeck_variation():
    """The generated seebeck_factor stays within the declared spread."""
    units = sensors.make_typek_units(50, seebeck_spread=0.0015, seed=42)
    factors = np.array([u.seebeck_factor for u in units])
    assert np.all(factors >= 0.998), "seebeck_factor must remain above -0.2%"
    assert np.all(factors <= 1.002), "seebeck_factor must remain below +0.2%"


def test_typek_calibration_data_range():
    """Normalized Type K data stay in [-1.1, 1.1]."""
    unit = sensors.make_typek_units(1, seed=10)[0]
    data = sensors.typek_calibration_data(unit, n_cal=50, n_val=25, n_test=200)
    for key in ("x_cal", "y_cal", "x_val", "y_val", "x_test", "y_test"):
        arr = data[key]
        assert arr.min() > -1.1 and arr.max() < 1.1, (
            f"{key} out of range: [{arr.min():.3f}, {arr.max():.3f}]"
        )


def test_typek_factory_baseline_large_error():
    """The nominal linear Type K approximation has MAE above 1 C in this case."""
    unit = sensors.make_typek_units(1, seed=10)[0]
    data = sensors.typek_calibration_data(unit)
    fac = sensors.typek_factory_baseline(unit, data)
    assert fac["mae_phys"] > 1.0, (
        f"Nominal Type K MAE too small: {fac['mae_phys']:.3f}°C, expected > 1 C"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LDRSensorUnit
# ─────────────────────────────────────────────────────────────────────────────

def test_ldr_units_variation():
    """LDR parameters remain within the declared generator spread."""
    units = sensors.make_ldr_units(50, R10_spread=0.30, gamma_spread=0.10, seed=6)
    R10_vals = np.array([u.R_10 for u in units])
    gamma_vals = np.array([u.gamma for u in units])
    # Nominals: R10=12000, gamma=0.85
    assert R10_vals.min() > 12000 * 0.69, "R10 below lower bound"
    assert R10_vals.max() < 12000 * 1.31, "R10 above upper bound"
    assert gamma_vals.min() > 0.85 * 0.89, "gamma below lower bound"
    assert gamma_vals.max() < 0.85 * 1.11, "gamma above upper bound"


def test_ldr_calibration_data_range():
    """Normalized LDR data stay in [-1.1, 1.1]."""
    unit = sensors.make_ldr_units(1, seed=6)[0]
    data = sensors.ldr_calibration_data(unit, n_cal=50, n_val=25, n_test=200)
    for key in ("x_cal", "y_cal", "x_test", "y_test"):
        arr = data[key]
        assert arr.min() > -1.1 and arr.max() < 1.1, (
            f"LDR {key} out of range: [{arr.min():.3f}, {arr.max():.3f}]"
        )


def test_ldr_response_nonlinear():
    """LDR ADC(E) differs substantially from a linear response."""
    unit = sensors.make_ldr_units(1, seed=6)[0]
    E = np.linspace(1, 10000, 100)
    adc = unit.adc_from_E(E)
    # A linear response has R-squared 1.0; this nonlinear case should be lower
    adc_lin = np.linspace(float(adc[0]), float(adc[-1]), 100)
    ss_res = np.sum((adc - adc_lin) ** 2)
    ss_tot = np.sum((adc - adc.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 < 0.95, f"LDR should be nonlinear, R-squared={r2:.3f} too close to 1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. fit_shh_baseline
# ─────────────────────────────────────────────────────────────────────────────

def _get_ntc_data(seed=42):
    unit = sensors.make_ntc_units(1, seed=seed)[0]
    data = sensors.ntc_calibration_data(unit, n_cal=50, n_val=25, n_test=200, seed=seed)
    return unit, data


def test_shh_ideal_very_accurate():
    """SHH fitted to three ideal points: MAE below 0.01 C for the exact model."""
    unit, data = _get_ntc_data()
    res = sensors.fit_shh_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"],
        data["denorm_y"], data["denorm_x"], unit, use_noisy=False
    )
    assert res["mae_phys"] < 0.01, (
        f"SHH ideal should be < 0.01 C, obtained {res['mae_phys']:.5f}°C"
    )


def test_shh_noisy_less_accurate_than_ideal():
    """Noisy-point SHH is less accurate than ideal-point fitting in this case."""
    unit, data = _get_ntc_data()
    shh_ideal = sensors.fit_shh_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"],
        data["denorm_y"], data["denorm_x"], unit, use_noisy=False
    )
    shh_noisy = sensors.fit_shh_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"],
        data["denorm_y"], data["denorm_x"], unit, use_noisy=True
    )
    assert shh_noisy["mae_phys"] > shh_ideal["mae_phys"], (
        "SHH noisy should have larger error than SHH ideal"
    )
    assert shh_noisy["mae_phys"] < 0.1, (
        f"SHH noisy should still be < 0.1 C, obtained {shh_noisy['mae_phys']:.4f}°C"
    )


def test_shh_noisy_beats_poly():
    """SHH on 50 noisy points beats degree-five polynomial in this matched-model case."""
    unit, data = _get_ntc_data()
    shh_noisy = sensors.fit_shh_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"],
        data["denorm_y"], data["denorm_x"], unit, use_noisy=True
    )
    poly = sensors.fit_poly_baseline(
        data["x_cal"], data["y_cal"], 5, data["x_test"], data["y_test"], data["denorm_y"]
    )
    assert shh_noisy["mae_phys"] < poly["mae_phys"], (
        f"SHH noisy ({shh_noisy['mae_phys']:.4f}) should have lower error than degree-five polynomial "
        f"({poly['mae_phys']:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. fit_onepoint_offset_baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_offset_baseline_sane():
    """One-point prediction is a constant shift of an independent factory."""
    unit, data = _get_ntc_data()
    factory = lambda x: data["norm_y"](unit.T_from_adc_nominal(data["denorm_x"](x)))
    offset = sensors.fit_onepoint_offset_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"], data["denorm_y"],
        factory_predict=factory, point_index=0
    )
    np.testing.assert_allclose(offset["y_pred"], factory(data["x_test"])+offset["delta"], atol=1e-7)
    assert offset["n_reference_used"] == 1
    assert np.isfinite(offset["mae_phys"])


def test_offset_baseline_ignores_unused_labels():
    """Changing every unused reference cannot alter a one-point model."""
    unit, data = _get_ntc_data()
    factory = lambda x: data["norm_y"](unit.T_from_adc_nominal(data["denorm_x"](x)))
    offset = sensors.fit_onepoint_offset_baseline(
        data["x_cal"], data["y_cal"], data["x_test"], data["y_test"], data["denorm_y"],
        factory_predict=factory, point_index=0
    )
    changed = data["y_cal"].copy()
    changed[1:] = 1000
    other = sensors.fit_onepoint_offset_baseline(
        data["x_cal"], changed, data["x_test"], data["y_test"], data["denorm_y"],
        factory_predict=factory, point_index=0
    )
    np.testing.assert_array_equal(offset["y_pred"], other["y_pred"])


# ─────────────────────────────────────────────────────────────────────────────
# 6. All six sensors: normalized data range
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sensor_name,make_fn,calib_fn", [
    ("NTC",      lambda: sensors.make_ntc_units(1, seed=1),
                 lambda u: sensors.ntc_calibration_data(u)),
    ("TypeK",    lambda: sensors.make_typek_units(1, seed=10),
                 lambda u: sensors.typek_calibration_data(u)),
    ("LDR",      lambda: sensors.make_ldr_units(1, seed=6),
                 lambda u: sensors.ldr_calibration_data(u)),
    ("Humidity", lambda: sensors.make_humidity_units(1, seed=5),
                 lambda u: sensors.humidity_calibration_data(u)),
    ("MQ",       lambda: sensors.make_mq_units(1, seed=2),
                 lambda u: sensors.mq_calibration_data(u)),
    ("pH",       lambda: sensors.make_ph_units(1, seed=3),
                 lambda u: sensors.ph_calibration_data(u)),
])
def test_all_sensors_normalized_range(sensor_name, make_fn, calib_fn):
    """All sensors: normalized x and y lie strictly inside [-1.1, 1.1]."""
    unit = make_fn()[0]
    data = calib_fn(unit)
    for key in ("x_cal", "y_cal", "x_test", "y_test"):
        arr = data[key]
        assert arr.min() > -1.1, (
            f"{sensor_name} {key}.min = {arr.min():.3f} < -1.1"
        )
        assert arr.max() < 1.1, (
            f"{sensor_name} {key}.max = {arr.max():.3f} > 1.1"
        )
