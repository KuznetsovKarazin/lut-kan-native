"""
sensors.py — Physical sensor simulation for TIM calibration study (v14).

Six sensor families:
  NTC thermistor       — Beta equation,        B ±3%       [real manufacturer data]
  Type K thermocouple  — NIST ITS-90 model,    Seebeck ±0.15%  [real NIST data]
  LDR photoresistor    — Power-law R(E),        gamma ±10%  [parametric, datasheet]
  MQ gas sensor        — Power-law Rs/R0,       A ±15%      [parametric]
  pH electrode         — Nernst equation,       S ±5%       [parametric]
  Humidity (resistive) — Exponential R(RH),     alpha ±15%  [parametric]

New in v14
----------
- Type K thermocouple with NIST ITS-90 polynomial model (real data reference)
- LDR photoresistor (logarithmic response — strong nonlinearity, LUT wins)
- fit_shh_baseline(): Steinhart-Hart fit on N noisy calibration points (fair comparison)
- fit_onepoint_offset_baseline(): single-point offset calibration baseline
- All baselines unified through a common interface
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable


# ─────────────────────────────────────────────────────────────────────────────
# Common ADC noise model
# ─────────────────────────────────────────────────────────────────────────────

def add_adc_noise(signal: np.ndarray, sigma: float,
                  rng: np.random.RandomState) -> np.ndarray:
    """Add Gaussian noise to simulate ADC quantisation + thermal noise."""
    if sigma <= 0:
        return signal.copy()
    return (signal + rng.randn(*signal.shape).astype(np.float32) * sigma).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NTC Thermistor  (Beta equation, B = 3950 ± 3%)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NTCSensorUnit:
    """One NTC thermistor unit with individual B-coefficient."""
    B: float
    R0: float = 10000
    T0: float = 298.15
    R_fix: float = 10000
    T_range: Tuple[float, float] = (0.0, 100.0)
    label: str = ""

    def adc_from_T(self, T_C: np.ndarray) -> np.ndarray:
        T_K = T_C + 273.15
        R_ntc = self.R0 * np.exp(self.B * (1.0 / T_K - 1.0 / self.T0))
        adc = 1023.0 * R_ntc / (self.R_fix + R_ntc)
        return adc.astype(np.float32)

    def T_from_adc_nominal(self, adc: np.ndarray, B_nom: float = 3950.0) -> np.ndarray:
        R_ntc = self.R_fix * adc / (1023.0 - adc + 1e-6)
        T_K = 1.0 / (1.0 / self.T0 + np.log(R_ntc / self.R0) / B_nom)
        return (T_K - 273.15).astype(np.float32)


def make_ntc_units(n_units: int = 8, B_nom: float = 3950.0,
                   B_spread: float = 0.03, seed: int = 1) -> List[NTCSensorUnit]:
    rng = np.random.RandomState(seed)
    B_values = B_nom * (1.0 + rng.uniform(-B_spread, B_spread, n_units))
    return [NTCSensorUnit(B=float(B), label=f"NTC_S{i+1}") for i, B in enumerate(B_values)]


def ntc_calibration_data(unit: NTCSensorUnit, n_cal=50, n_val=25, n_test=200,
                          adc_noise_sigma=1.0, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    T_min, T_max = unit.T_range
    T_cal  = rng.uniform(T_min, T_max, n_cal).astype(np.float32)
    T_val  = rng.uniform(T_min, T_max, n_val).astype(np.float32)
    T_test = np.linspace(T_min, T_max, n_test).astype(np.float32)
    adc_cal  = add_adc_noise(unit.adc_from_T(T_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_T(T_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_T(T_test)
    adc_at_Tmin = unit.adc_from_T(np.array([float(T_min)]))[0]
    adc_at_Tmax = unit.adc_from_T(np.array([float(T_max)]))[0]
    adc_lo = float(min(adc_at_Tmin, adc_at_Tmax))
    adc_hi = float(max(adc_at_Tmin, adc_at_Tmax))
    adc_mid  = (adc_hi + adc_lo) / 2.0
    adc_half = (adc_hi - adc_lo) / 2.0 * 1.05
    def norm_x(adc): return ((adc - adc_mid) / adc_half).astype(np.float32)
    def denorm_x(xn): return (xn * adc_half + adc_mid).astype(np.float32)
    T_mid = (T_max + T_min) / 2.0
    T_half = (T_max - T_min) / 2.0
    def norm_y(T): return ((T - T_mid) / T_half).astype(np.float32)
    def denorm_y(yn): return (yn * T_half + T_mid).astype(np.float32)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(T_cal),
        x_val=norm_x(adc_val), y_val=norm_y(T_val),
        x_test=norm_x(adc_test), y_test=norm_y(T_test),
        T_test=T_test, denorm_y=denorm_y, norm_x=norm_x, denorm_x=denorm_x,
        unit=unit, sensor_type="ntc", physical_unit="°C", T_range=unit.T_range,
    )


def ntc_factory_baseline(unit: NTCSensorUnit, data: dict) -> dict:
    adc_test = data["denorm_x"](data["x_test"])
    T_pred = unit.T_from_adc_nominal(adc_test).astype(np.float32)
    T_true = data["T_test"]
    mae_phys = float(np.mean(np.abs(T_pred - T_true)))
    T_half = (unit.T_range[1] - unit.T_range[0]) / 2.0
    mse_norm = float(np.mean(((T_pred - T_true) / T_half) ** 2))
    return dict(T_pred=T_pred, mae_phys=mae_phys, mse_norm=mse_norm)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Type K Thermocouple  (NIST ITS-90 polynomial, Seebeck ±0.15%)
# ─────────────────────────────────────────────────────────────────────────────
# NIST ITS-90 polynomial coefficients for Type K EMF(T) in mV
# Source: https://srdata.nist.gov/its90/type_k/kcoefficients.html
# Two sub-ranges: range 1: -270 to 0°C (11 coefficients)
#                 range 2:    0 to 1372°C (10 coefficients + exponential term)

_K_C1 = np.array([0.000000000000e+00,  3.945012802250e-02,  2.362237359230e-05,
                  -3.285890678233e-07, -4.990482877342e-09, -6.750905917702e-11,
                  -5.741032742359e-13, -3.108887289069e-15, -1.045160936020e-17,
                  -1.988926687100e-20, -1.632095645973e-23], dtype=np.float64)

_K_C2 = np.array([-1.760041368621e-02,  3.892120497416e-02,  1.855877003098e-05,
                  -9.945759287054e-08,  3.184094571302e-10, -5.607505339562e-13,
                   5.607522819265e-16, -3.202072000550e-19,  9.715114715606e-23,
                  -1.210472259908e-26], dtype=np.float64)

# Exponential correction term for range 2 (0 to 1372°C)
_K_A0 = 0.1185976  # mV
_K_A1 = -1.183432e-04  # 1/°C²
_K_A2 = 126.9686  # °C


def nist_k_emf(T_C: np.ndarray) -> np.ndarray:
    """
    NIST ITS-90 Type K thermocouple: Temperature [°C] → EMF [mV].
    Valid range: -270 to 1372°C.  Two sub-ranges with polynomial + exp term.
    """
    T = np.asarray(T_C, dtype=np.float64)
    emf = np.zeros_like(T)
    # Range 1: -270 to 0°C
    m1 = T < 0.0
    if m1.any():
        T1 = T[m1]
        e = np.zeros_like(T1)
        for i, c in enumerate(_K_C1):
            e += c * T1**i
        emf[m1] = e
    # Range 2: 0 to 1372°C
    m2 = T >= 0.0
    if m2.any():
        T2 = T[m2]
        e = np.zeros_like(T2)
        for i, c in enumerate(_K_C2):
            e += c * T2**i
        e += _K_A0 * np.exp(_K_A1 * (T2 - _K_A2)**2)
        emf[m2] = e
    return emf.astype(np.float32)


@dataclass
class TypeKThermocoupleUnit:
    """
    Type K thermocouple with unit-to-unit Seebeck coefficient variation.

    Physical model (positive-range amplifier circuit):
        EMF(T)  = NIST_ITS90_K(T) * seebeck_factor            [mV]
        V_amp   = EMF * amp_gain * gain_factor * 1e-3 + V_bias [V]
        ADC     = clip(V_amp / V_ref * adc_max, 0, adc_max)

    V_bias = 0.27 V shifts the output so T=0°C → ADC ≈ 335 counts,
    avoiding the ADC floor and keeping the mapping strictly injective
    over T_range = (0, 350)°C.  This models a real cold-junction-
    compensated amplifier (e.g. AD8495 / MAX31855 breakout-board style).

    Batch variation:
        seebeck_factor  ±0.15%  (IEC 60584-1 class 1 tolerance)
        gain_factor     ±0.5%   (op-amp gain resistor tolerance)

    Factory firmware: linear fit using nominal Seebeck ~41 μV/°C.

    NOTE (v14 fix): the previous T_range = (-50, 350) caused V_amp < 0
    for T < 0°C, which np.clip mapped to ADC = 0, making T(ADC) non-
    injective.  T_range is now (0, 350), matching the positive-EMF
    region of the NIST ITS-90 table.
    """
    seebeck_factor: float = 1.0     # multiplicative on EMF (nominal = 1.0)
    gain_factor: float = 1.0        # amplifier gain resistor variation
    T_range: Tuple[float, float] = (0.0, 350.0)   # positive-EMF range (v14 fix)
    V_ref: float = 3.3              # ADC reference voltage [V]
    amp_gain: float = 100.0         # nominal amplifier gain (mV → V)
    V_bias: float = 0.27            # amplifier output offset [V] at T=0°C
    ADC_bits: int = 12
    label: str = ""

    def emf_from_T(self, T_C: np.ndarray) -> np.ndarray:
        """EMF [mV] including unit-specific Seebeck variation."""
        return nist_k_emf(T_C) * self.seebeck_factor

    def adc_from_T(self, T_C: np.ndarray) -> np.ndarray:
        """ADC reading (0 to 2^ADC_bits - 1) from temperature.

        V_amp = EMF_mV * amp_gain * gain_factor * 1e-3 + V_bias
        Strictly monotone and positive over T_range = (0, 350)°C.
        """
        emf_mV = self.emf_from_T(T_C)
        V_amp = emf_mV * 1e-3 * self.amp_gain * self.gain_factor + self.V_bias
        adc_max = float(2**self.ADC_bits - 1)
        adc = np.clip(V_amp / self.V_ref * adc_max, 0, adc_max)
        return adc.astype(np.float32)

    def T_from_adc_linear(self, adc: np.ndarray) -> np.ndarray:
        """Factory linear calibration: constant Seebeck ~41 μV/°C + bias offset."""
        adc_max = float(2**self.ADC_bits - 1)
        V_amp = adc / adc_max * self.V_ref
        emf_mV_nominal = (V_amp - self.V_bias) / (self.amp_gain * 1e-3)
        SEEBECK_NOM_mV_per_C = 0.04096   # mV/°C average 0–400°C for Type K
        T_C = emf_mV_nominal / SEEBECK_NOM_mV_per_C
        return T_C.astype(np.float32)


def make_typek_units(n_units: int = 8,
                     seebeck_spread: float = 0.0015,
                     gain_spread: float = 0.005,
                     seed: int = 10) -> List[TypeKThermocoupleUnit]:
    """
    Generate n_units Type K thermocouples.
    seebeck_spread = 0.15% matches IEC 60584-1 class 1 tolerance.
    gain_spread = 0.5% typical op-amp resistor tolerance.
    """
    rng = np.random.RandomState(seed)
    sk = 1.0 + rng.uniform(-seebeck_spread, seebeck_spread, n_units)
    gf = 1.0 + rng.uniform(-gain_spread, gain_spread, n_units)
    return [TypeKThermocoupleUnit(seebeck_factor=float(s), gain_factor=float(g),
                                   label=f"K_S{i+1}")
            for i, (s, g) in enumerate(zip(sk, gf))]


def typek_calibration_data(unit: TypeKThermocoupleUnit, n_cal=50, n_val=25,
                             n_test=200, adc_noise_sigma=2.0, seed=42) -> dict:
    """
    Calibration data for Type K thermocouple.
    x = normalised ADC,  y = normalised temperature.
    adc_noise_sigma=2.0 ≈ 0.5 mV noise on 12-bit 3.3V ADC (realistic).
    """
    rng = np.random.RandomState(seed)
    T_min, T_max = unit.T_range
    T_cal  = rng.uniform(T_min, T_max, n_cal).astype(np.float32)
    T_val  = rng.uniform(T_min, T_max, n_val).astype(np.float32)
    T_test = np.linspace(T_min, T_max, n_test).astype(np.float32)
    adc_cal  = add_adc_noise(unit.adc_from_T(T_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_T(T_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_T(T_test)
    # Normalise using actual sensor range
    adc_lo = float(adc_test.min()); adc_hi = float(adc_test.max())
    adc_mid  = (adc_hi + adc_lo) / 2.0
    adc_half = (adc_hi - adc_lo) / 2.0 * 1.05
    def norm_x(adc): return ((adc - adc_mid) / adc_half).astype(np.float32)
    def denorm_x(xn): return (xn * adc_half + adc_mid).astype(np.float32)
    T_mid = (T_max + T_min) / 2.0
    T_half = (T_max - T_min) / 2.0
    def norm_y(T): return ((T - T_mid) / T_half).astype(np.float32)
    def denorm_y(yn): return (yn * T_half + T_mid).astype(np.float32)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(T_cal),
        x_val=norm_x(adc_val), y_val=norm_y(T_val),
        x_test=norm_x(adc_test), y_test=norm_y(T_test),
        T_test=T_test, denorm_y=denorm_y, norm_x=norm_x, denorm_x=denorm_x,
        unit=unit, sensor_type="typek", physical_unit="°C", T_range=unit.T_range,
    )


def typek_factory_baseline(unit: TypeKThermocoupleUnit, data: dict) -> dict:
    """Linear (constant Seebeck) factory calibration."""
    adc_test = data["denorm_x"](data["x_test"])
    T_pred = unit.T_from_adc_linear(adc_test)
    T_true = data["T_test"]
    mae_phys = float(np.mean(np.abs(T_pred - T_true)))
    return dict(T_pred=T_pred, mae_phys=mae_phys)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LDR Photoresistor  (GL55-series, power-law R(E), gamma ±10%)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LDRSensorUnit:
    """
    Light-dependent resistor (photoresistor).
    Model: R = R_10 * (10 / E)^gamma
    where:
        R_10  : resistance at 10 lux [Ω]
        gamma : slope exponent, ≈0.85 for GL5516 (0.7–1.0 range)
        E     : illuminance [lux]

    ADC from voltage divider: ADC = 1023 * R_fixed / (R_LDR + R_fixed)

    Batch variation:
        R_10  ±30%   (GL55-series datasheet: min/max ratio ~3:1)
        gamma ±10%   (from characteristic curves in datasheet)
    """
    R_10: float           # resistance at 10 lux [Ω]
    gamma: float          # slope exponent
    R_fixed: float = 10000  # fixed resistor in divider [Ω]
    E_range: Tuple[float, float] = (1.0, 10000.0)  # lux
    label: str = ""

    def adc_from_E(self, E_lux: np.ndarray) -> np.ndarray:
        """ADC output (0–1023) from illuminance [lux]."""
        R_ldr = self.R_10 * (10.0 / np.maximum(E_lux, 1e-3)) ** self.gamma
        adc = 1023.0 * self.R_fixed / (R_ldr + self.R_fixed)
        return np.clip(adc, 0, 1023).astype(np.float32)

    def E_from_adc_linear(self, adc: np.ndarray) -> np.ndarray:
        """Factory linear baseline: assumes E ∝ ADC (very poor approximation)."""
        E_min, E_max = self.E_range
        return (adc / 1023.0 * (E_max - E_min) + E_min).astype(np.float32)


def make_ldr_units(n_units: int = 8, R10_nom: float = 12000.0,
                   R10_spread: float = 0.30, gamma_nom: float = 0.85,
                   gamma_spread: float = 0.10, seed: int = 6) -> List[LDRSensorUnit]:
    """GL5516 typical: R10 = 10–20 kΩ, gamma = 0.7–1.0."""
    rng = np.random.RandomState(seed)
    R10_vals   = R10_nom  * (1.0 + rng.uniform(-R10_spread,   R10_spread,   n_units))
    gamma_vals = gamma_nom * (1.0 + rng.uniform(-gamma_spread, gamma_spread, n_units))
    return [LDRSensorUnit(R_10=float(r), gamma=float(g), label=f"LDR_S{i+1}")
            for i, (r, g) in enumerate(zip(R10_vals, gamma_vals))]


def ldr_calibration_data(unit: LDRSensorUnit, n_cal=50, n_val=25, n_test=200,
                          adc_noise_sigma=1.0, seed=42) -> dict:
    """
    Calibration data for LDR.
    x = normalised ADC,  y = normalised log10(E_lux).
    Log scale for target is natural for logarithmic sensors.
    """
    rng = np.random.RandomState(seed)
    E_min, E_max = unit.E_range
    logE_min, logE_max = np.log10(E_min), np.log10(E_max)
    logE_cal  = rng.uniform(logE_min, logE_max, n_cal).astype(np.float32)
    logE_val  = rng.uniform(logE_min, logE_max, n_val).astype(np.float32)
    logE_test = np.linspace(logE_min, logE_max, n_test).astype(np.float32)
    E_cal, E_val = 10.0**logE_cal, 10.0**logE_val
    E_test = 10.0**logE_test
    adc_cal  = add_adc_noise(unit.adc_from_E(E_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_E(E_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_E(E_test)
    adc_lo = float(adc_test.min()); adc_hi = float(adc_test.max())
    adc_mid  = (adc_hi + adc_lo) / 2.0
    adc_half = (adc_hi - adc_lo) / 2.0 * 1.05
    def norm_x(adc): return ((adc - adc_mid) / adc_half).astype(np.float32)
    def denorm_x(xn): return (xn * adc_half + adc_mid).astype(np.float32)
    logE_mid  = (logE_max + logE_min) / 2.0
    logE_half = (logE_max - logE_min) / 2.0
    def norm_y(le): return ((le - logE_mid) / logE_half).astype(np.float32)
    def denorm_y(yn): return (yn * logE_half + logE_mid).astype(np.float32)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(logE_cal),
        x_val=norm_x(adc_val), y_val=norm_y(logE_val),
        x_test=norm_x(adc_test), y_test=norm_y(logE_test),
        E_test=E_test, logE_test=logE_test,
        denorm_y=denorm_y, norm_x=norm_x, denorm_x=denorm_x,
        unit=unit, sensor_type="ldr", physical_unit="log10(lux)", E_range=unit.E_range,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MQ Gas Sensor  (power-law, ±15% sensitivity)  — unchanged from v13
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MQSensorUnit:
    A: float; B_exp: float
    R0: float = 10000; R_load: float = 5000
    C_range: Tuple[float, float] = (100.0, 10000.0)
    label: str = ""

    def adc_from_C(self, C_ppm: np.ndarray) -> np.ndarray:
        Rs_over_R0 = self.A * (C_ppm ** (-self.B_exp))
        Rs = Rs_over_R0 * self.R0
        adc = 1023.0 * self.R_load / (Rs + self.R_load)
        return np.clip(adc, 0, 1023).astype(np.float32)


def make_mq_units(n_units=8, A_nom=2.3, A_spread=0.15,
                  B_nom=0.36, B_spread=0.05, seed=2) -> List[MQSensorUnit]:
    rng = np.random.RandomState(seed)
    A_vals = A_nom * (1.0 + rng.uniform(-A_spread, A_spread, n_units))
    B_vals = B_nom * (1.0 + rng.uniform(-B_spread, B_spread, n_units))
    return [MQSensorUnit(A=float(a), B_exp=float(b), label=f"MQ_S{i+1}")
            for i, (a, b) in enumerate(zip(A_vals, B_vals))]


def mq_calibration_data(unit: MQSensorUnit, n_cal=50, n_val=25, n_test=200,
                         adc_noise_sigma=1.5, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    C_min, C_max = unit.C_range
    logC_min, logC_max = np.log10(C_min), np.log10(C_max)
    logC_cal  = rng.uniform(logC_min, logC_max, n_cal).astype(np.float32)
    logC_val  = rng.uniform(logC_min, logC_max, n_val).astype(np.float32)
    logC_test = np.linspace(logC_min, logC_max, n_test).astype(np.float32)
    C_cal, C_val = 10.0**logC_cal, 10.0**logC_val
    C_test = 10.0**logC_test
    adc_cal  = add_adc_noise(unit.adc_from_C(C_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_C(C_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_C(C_test)
    adc_at_Cmin = unit.adc_from_C(np.array([float(C_min)]))[0]
    adc_at_Cmax = unit.adc_from_C(np.array([float(C_max)]))[0]
    adc_lo_mq = float(min(adc_at_Cmin, adc_at_Cmax))
    adc_hi_mq = float(max(adc_at_Cmin, adc_at_Cmax))
    adc_mid_mq  = (adc_hi_mq + adc_lo_mq) / 2.0
    adc_half_mq = (adc_hi_mq - adc_lo_mq) / 2.0 * 1.05
    def norm_x(adc): return ((adc - adc_mid_mq) / adc_half_mq).astype(np.float32)
    def denorm_x(xn): return (xn * adc_half_mq + adc_mid_mq).astype(np.float32)
    logC_mid  = (logC_max + logC_min) / 2.0
    logC_half = (logC_max - logC_min) / 2.0
    def norm_y(lc): return ((lc - logC_mid) / logC_half).astype(np.float32)
    def denorm_y(yn): return (yn * logC_half + logC_mid).astype(np.float32)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(logC_cal),
        x_val=norm_x(adc_val), y_val=norm_y(logC_val),
        x_test=norm_x(adc_test), y_test=norm_y(logC_test),
        C_test=C_test, logC_test=logC_test,
        denorm_y=denorm_y, norm_x=norm_x, denorm_x=denorm_x,
        unit=unit, sensor_type="mq", physical_unit="log10(ppm)", C_range=unit.C_range,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. pH Electrode  (Nernst, slope ±5%)  — unchanged from v13
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PHElectrodeUnit:
    E0: float; S: float
    pH_range: Tuple[float, float] = (2.0, 12.0)
    E_full_scale: float = 414.0
    ADC_bits: int = 12; label: str = ""

    def adc_from_pH(self, pH: np.ndarray) -> np.ndarray:
        E_mV = self.E0 + self.S * (7.0 - pH)
        adc_max = float(2**self.ADC_bits - 1)
        adc = adc_max * (E_mV + self.E_full_scale) / (2.0 * self.E_full_scale)
        return np.clip(adc, 0, adc_max).astype(np.float32)


def make_ph_units(n_units=8, S_nom=59.16, S_spread=0.05,
                  E0_spread_mV=10.0, seed=3) -> List[PHElectrodeUnit]:
    rng = np.random.RandomState(seed)
    S_vals  = S_nom * (1.0 + rng.uniform(-S_spread, S_spread, n_units))
    E0_vals = rng.uniform(-E0_spread_mV, E0_spread_mV, n_units)
    return [PHElectrodeUnit(E0=float(e0), S=float(s), label=f"pH_S{i+1}")
            for i, (e0, s) in enumerate(zip(E0_vals, S_vals))]


def ph_calibration_data(unit: PHElectrodeUnit, n_cal=50, n_val=25, n_test=200,
                         adc_noise_sigma=2.0, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    pH_min, pH_max = unit.pH_range
    pH_cal  = rng.uniform(pH_min, pH_max, n_cal).astype(np.float32)
    pH_val  = rng.uniform(pH_min, pH_max, n_val).astype(np.float32)
    pH_test = np.linspace(pH_min, pH_max, n_test).astype(np.float32)
    adc_cal  = add_adc_noise(unit.adc_from_pH(pH_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_pH(pH_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_pH(pH_test)
    adc_max = float(2**unit.ADC_bits - 1)
    def norm_x(adc): return (adc / (adc_max / 2.0) - 1.0).astype(np.float32)
    pH_mid  = (pH_max + pH_min) / 2.0
    pH_half = (pH_max - pH_min) / 2.0
    def norm_y(ph): return ((ph - pH_mid) / pH_half).astype(np.float32)
    def denorm_y(yn): return (yn * pH_half + pH_mid).astype(np.float32)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(pH_cal),
        x_val=norm_x(adc_val), y_val=norm_y(pH_val),
        x_test=norm_x(adc_test), y_test=norm_y(pH_test),
        pH_test=pH_test, denorm_y=denorm_y, norm_x=norm_x,
        unit=unit, sensor_type="ph", physical_unit="pH units", pH_range=unit.pH_range,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Resistive Humidity sensor  — unchanged from v13
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HumiditySensorUnit:
    alpha: float; R0: float
    R_fixed: float = 100_000; Vcc: float = 3.3
    RH_range: Tuple[float, float] = (10.0, 90.0)
    ADC_bits: int = 12; label: str = ""

    def adc_from_RH(self, RH: np.ndarray) -> np.ndarray:
        adc_max = float(2**self.ADC_bits - 1)
        R = self.R0 * np.exp(-self.alpha * RH)
        V = self.Vcc * self.R_fixed / (R + self.R_fixed)
        adc = V / self.Vcc * adc_max
        return np.clip(adc, 0, adc_max).astype(np.float32)


def make_humidity_units(n_units=8, alpha_nom=0.06, alpha_spread=0.15,
                        R0_nom=1_000_000, R0_spread=0.10, seed=5) -> List[HumiditySensorUnit]:
    rng = np.random.RandomState(seed)
    alpha_vals = alpha_nom * (1.0 + rng.uniform(-alpha_spread, alpha_spread, n_units))
    R0_vals    = R0_nom   * (1.0 + rng.uniform(-R0_spread, R0_spread, n_units))
    return [HumiditySensorUnit(alpha=float(a), R0=float(r), label=f"HUM_S{i+1}")
            for i, (a, r) in enumerate(zip(alpha_vals, R0_vals))]


def humidity_calibration_data(unit: HumiditySensorUnit, n_cal=50, n_val=25,
                               n_test=200, adc_noise_sigma=2.0, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    RH_min, RH_max = unit.RH_range
    RH_cal  = rng.uniform(RH_min, RH_max, n_cal).astype(np.float32)
    RH_val  = rng.uniform(RH_min, RH_max, n_val).astype(np.float32)
    RH_test = np.linspace(RH_min, RH_max, n_test).astype(np.float32)
    adc_cal  = add_adc_noise(unit.adc_from_RH(RH_cal),  adc_noise_sigma, rng)
    adc_val  = add_adc_noise(unit.adc_from_RH(RH_val),  adc_noise_sigma, rng)
    adc_test = unit.adc_from_RH(RH_test)
    adc_lo = float(adc_test.min()); adc_hi = float(adc_test.max())
    adc_mid  = (adc_hi + adc_lo) / 2.0
    adc_half = (adc_hi - adc_lo) / 2.0 * 1.05
    def norm_x(adc): return ((adc - adc_mid) / adc_half).astype(np.float32)
    def denorm_x(xn): return (xn * adc_half + adc_mid).astype(np.float32)
    RH_mid = (RH_max + RH_min) / 2.0
    RH_half = (RH_max - RH_min) / 2.0
    def norm_y(rh): return ((rh - RH_mid) / RH_half).astype(np.float32)
    def denorm_y(yn): return (yn * RH_half + RH_mid).astype(np.float32)
    nominal = HumiditySensorUnit(alpha=0.06, R0=1_000_000,
                                  R_fixed=unit.R_fixed, RH_range=unit.RH_range,
                                  ADC_bits=unit.ADC_bits)
    return dict(
        x_cal=norm_x(adc_cal), y_cal=norm_y(RH_cal),
        x_val=norm_x(adc_val), y_val=norm_y(RH_val),
        x_test=norm_x(adc_test), y_test=norm_y(RH_test),
        RH_test=RH_test, denorm_y=denorm_y, norm_x=norm_x, denorm_x=denorm_x,
        nominal_unit=nominal, unit=unit,
        sensor_type="humidity", physical_unit="%RH", RH_range=unit.RH_range,
    )


def humidity_factory_baseline(unit: HumiditySensorUnit, data: dict) -> dict:
    adc_max = float(2**unit.ADC_bits - 1)
    adc_test = data["denorm_x"](data["x_test"])
    V = adc_test / adc_max * unit.Vcc
    R = unit.R_fixed * (unit.Vcc - V) / (V + 1e-9)
    RH_pred = -np.log(np.maximum(R, 1.0) / 1_000_000) / 0.06
    RH_pred = np.clip(RH_pred, *unit.RH_range).astype(np.float32)
    mae_phys = float(np.mean(np.abs(RH_pred - data["RH_test"])))
    return dict(RH_pred=RH_pred, mae_phys=mae_phys)


# ─────────────────────────────────────────────────────────────────────────────
# Shared calibration baselines
# ─────────────────────────────────────────────────────────────────────────────

def fit_poly_baseline(x_cal, y_cal, degree: int, x_test, y_test,
                      denorm_y=None) -> dict:
    """Polynomial fit on calibration data; MAE on test set."""
    coeffs = np.polyfit(x_cal.astype(float), y_cal.astype(float), degree)
    y_pred = np.polyval(coeffs, x_test.astype(float)).astype(np.float32)
    mse = float(np.mean((y_pred - y_test) ** 2))
    mae_phys = float(np.mean(np.abs(denorm_y(y_pred) - denorm_y(y_test)))) \
               if denorm_y is not None else None
    return dict(coeffs=coeffs, y_pred=y_pred, mse=mse, mae_phys=mae_phys)


def fit_onepoint_offset_baseline(x_cal, y_cal, x_test, y_test,
                                  denorm_y=None, poly_degree: int = 5) -> dict:
    """
    Single-point offset calibration (v14 new baseline).

    Represents the simplest field calibration used in practice:
        1. Factory polynomial predicts T_factory(ADC)
        2. One reference measurement gives δ = T_ref - T_factory(ADC_ref)
        3. Correction: T_cal(ADC) = T_factory(ADC) + δ

    Implementation: fit the factory polynomial on calibration data,
    compute the median residual δ = median(y_cal - y_factory(x_cal)),
    shift all test predictions by δ.  This approximates single-point
    offset without requiring explicit access to the factory model parameters.
    """
    # Step 1: fit factory polynomial on all calibration points
    coeffs = np.polyfit(x_cal.astype(float), y_cal.astype(float), poly_degree)
    y_factory_cal  = np.polyval(coeffs, x_cal.astype(float)).astype(np.float32)
    y_factory_test = np.polyval(coeffs, x_test.astype(float)).astype(np.float32)

    # Step 2: estimate constant offset from a single midpoint measurement
    # (simulate as median of all calibration residuals = best-case single point)
    residuals = y_cal - y_factory_cal
    delta = float(np.median(residuals))

    # Step 3: shift factory prediction by delta
    y_pred = (y_factory_test + delta).astype(np.float32)
    mse = float(np.mean((y_pred - y_test) ** 2))
    mae_phys = float(np.mean(np.abs(denorm_y(y_pred) - denorm_y(y_test)))) \
               if denorm_y is not None else None
    return dict(delta=delta, coeffs=coeffs, y_pred=y_pred, mse=mse, mae_phys=mae_phys)


def fit_shh_baseline(x_cal, y_cal, x_test, y_test, denorm_y,
                      denorm_x, unit: NTCSensorUnit,
                      n_points: int = 3,
                      use_noisy: bool = False) -> dict:
    """
    Steinhart-Hart three-parameter fit for NTC (v14 fair comparison).

    Two modes:
        use_noisy=False  : SHH fit from 3 ideal (noiseless) points evenly
                           spaced across T_range — the standard literature setup.
        use_noisy=True   : SHH fit from the same n_cal noisy calibration points
                           (via least-squares) — the fair comparison with LUT.

    Model: 1/T = A + B*ln(R) + C*(ln(R))^3
    """
    T_min, T_max = unit.T_range
    if not use_noisy:
        # Standard 3-point ideal fit
        T_ref = np.linspace(T_min, T_max, n_points)
        adc_ref = unit.adc_from_T(T_ref)  # noiseless
    else:
        # Same noisy calibration points used by LUT
        adc_ref = denorm_x(x_cal)
        T_ref   = denorm_y(y_cal)

    R_ref = unit.R_fix * adc_ref / (1023.0 - adc_ref + 1e-9)
    lnR   = np.log(np.maximum(R_ref, 1e-9))
    T_K   = T_ref + 273.15

    # Least-squares: [1, lnR, (lnR)^3] @ [A, B, C]^T = 1/T_K
    X = np.column_stack([np.ones_like(lnR), lnR, lnR**3])
    y_inv = 1.0 / T_K
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_inv, rcond=None)
    except np.linalg.LinAlgError:
        return dict(mae_phys=np.inf, mse=np.inf, coeffs=None)

    A, B, C = coeffs
    # Evaluate on test set
    adc_test = denorm_x(x_test)
    R_test   = unit.R_fix * adc_test / (1023.0 - adc_test + 1e-9)
    lnR_test = np.log(np.maximum(R_test, 1e-9))
    T_K_pred = 1.0 / (A + B * lnR_test + C * lnR_test**3)
    T_pred   = (T_K_pred - 273.15).astype(np.float32)
    T_true   = denorm_y(y_test)
    mae_phys = float(np.mean(np.abs(T_pred - T_true)))
    mse      = float(np.mean(((T_pred - T_true) / ((T_max - T_min) / 2.0)) ** 2))
    return dict(coeffs=(A, B, C), T_pred=T_pred, mae_phys=mae_phys, mse=mse)


def ntc_factory_baseline(unit: NTCSensorUnit, data: dict) -> dict:
    adc_test = data["denorm_x"](data["x_test"])
    T_pred = unit.T_from_adc_nominal(adc_test).astype(np.float32)
    T_true = data["T_test"]
    mae_phys = float(np.mean(np.abs(T_pred - T_true)))
    T_half = (unit.T_range[1] - unit.T_range[0]) / 2.0
    mse_norm = float(np.mean(((T_pred - T_true) / T_half) ** 2))
    return dict(T_pred=T_pred, mae_phys=mae_phys, mse_norm=mse_norm)
