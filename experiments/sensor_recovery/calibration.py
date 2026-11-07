"""Auditable scalar calibration models; NumPy/SciPy only.

This module intentionally does not import lut_native's PyTorch package. The
new pilot is independent of historical results and makes no MCU-time claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
from scipy.interpolate import BSpline

LAMBDAS = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
DEGREES = (1, 2, 3, 4, 5, 7, 9)
COMMON_BYTES = 16  # x_mid, x_half, y_mid, y_half, each float32


def linear_basis(x, length):
    """Endpoint-inclusive hat functions, with identical train/deploy indexing."""
    if length < 2:
        raise ValueError("A LUT needs at least two nodes")
    x = np.asarray(x, dtype=float).reshape(-1)
    p = (np.clip(x, -1, 1) + 1) * ((length - 1) / 2)
    lo = np.minimum(p.astype(int), length - 2)
    w = p - lo
    a = np.zeros((len(x), length))
    a[np.arange(len(x)), lo] = 1 - w
    a[np.arange(len(x)), lo + 1] = w
    return a


def spline_knots(n_coeff, degree=3):
    if n_coeff <= degree:
        raise ValueError("Too few spline coefficients")
    breaks = np.linspace(-1, 1, n_coeff - degree + 1)
    return np.r_[np.repeat(-1.0, degree), breaks, np.repeat(1.0, degree)]


def cubic_basis(x, n_coeff, silu=False):
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), -1, 1)
    a = BSpline(spline_knots(n_coeff), np.eye(n_coeff), 3)(x)
    if silu:
        a = np.column_stack([a, x / (1 + np.exp(-x))])
    return a


def regularized_fit(a, y, lam, n_smooth=None):
    """Minimize mean squared residual + lambda * mean squared second diff.

    A KAN SiLU coefficient is deliberately excluded from the spline penalty.
    Augmented least squares avoids squaring the condition number.
    """
    a, y = np.asarray(a, float), np.asarray(y, float)
    if a.ndim != 2 or len(y) != len(a) or not len(y):
        raise ValueError("Invalid training arrays")
    n_smooth = a.shape[1] if n_smooth is None else n_smooth
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    if lam > 0 and n_smooth > 2:
        d = np.diff(np.eye(n_smooth), n=2, axis=0)
        d = np.pad(d, ((0, 0), (0, a.shape[1] - n_smooth)))
        design = np.vstack([a / np.sqrt(len(y)), np.sqrt(lam / len(d)) * d])
        target = np.r_[y / np.sqrt(len(y)), np.zeros(len(d))]
    else:
        design, target = a, y
    return np.linalg.lstsq(design, target, rcond=1e-12)[0]


def objective(a, y, theta, lam):
    return float(np.mean((a @ theta - y) ** 2) +
                 lam * np.mean(np.diff(theta, n=2) ** 2))


def adam_fit(a, y, lam=1.0, epochs=1000, lr=0.01):
    """Deterministic full-batch optimizer diagnostic, not legacy Adam replay."""
    theta = np.zeros(a.shape[1])
    m, v = theta.copy(), theta.copy()
    d = np.diff(np.eye(len(theta)), n=2, axis=0)
    gram, rhs = a.T @ a / len(y), a.T @ y / len(y)
    gram += lam * d.T @ d / len(d)
    for step in range(1, epochs + 1):
        g = 2 * (gram @ theta - rhs)
        m = .9 * m + .1 * g
        v = .999 * v + .001 * g * g
        theta -= lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
    return theta


@dataclass
class Transform:
    x_mid: float
    x_half: float
    y_mid: float
    y_half: float

    @classmethod
    def from_references(cls, x, q_range):
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            raise ValueError("References do not span an input range")
        # Cast exactly the metadata that the deployment serializer will store.
        return cls(*map(float, np.asarray([(lo + hi) / 2, (hi - lo) * .525,
                                          np.mean(q_range), np.ptp(q_range) / 2],
                                         dtype=np.float32)))

    def nx(self, x):
        return (np.asarray(x) - self.x_mid) / self.x_half

    def dx(self, x):
        return np.asarray(x) * self.x_half + self.x_mid

    def ny(self, y):
        return (np.asarray(y) - self.y_mid) / self.y_half

    def dy(self, y):
        return np.asarray(y) * self.y_half + self.y_mid


@dataclass
class Sensor:
    kind: str
    unit: int
    parameters: dict

    @property
    def q_range(self):
        return (0.0, 3.0) if self.kind == "ldr" else (0.0, 100.0)

    @property
    def quantity(self):
        return "log10(lux)" if self.kind == "ldr" else "degC"

    def adc(self, q):
        q = np.asarray(q, float)
        p = self.parameters
        if self.kind == "ldr":
            resistance = p["r0"] * (10 / (10 ** q)) ** p["gamma"]
        else:
            resistance = p["r0"] * np.exp(p["beta"] * (1 / (q + 273.15) - 1 / 298.15))
        adc = resistance / (10000 + resistance)
        if self.kind == "ntc_frontend":
            adc = adc + 4 / 1023 * np.sin(6 * np.pi * adc + p["phase"])
        return adc

    def observe(self, q, rng, averages=16):
        codes = 1023 * self.adc(q)
        samples = np.clip(np.rint(codes[:, None] + rng.normal(0, 1, (len(codes), averages))), 0, 1023)
        return samples.mean(axis=1) / 1023

    def factory(self, adc):
        adc = np.clip(adc, 1e-7, 1 - 1e-7)
        ratio = adc / (1 - adc)
        if self.kind == "ldr":
            return 1 - np.log10(ratio) / .7
        return 1 / (1 / 298.15 + np.log(ratio) / 3950) - 273.15


def make_sensor(kind, unit, root_seed=20261107):
    if kind not in ("ntc", "ntc_frontend", "ldr"):
        raise ValueError(kind)
    # Matched NTC and frontend-stress units have identical physical parameters.
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, unit, int(kind == "ldr")]))
    if kind == "ldr":
        params = {"r0": float(10000 * rng.uniform(.7, 1.3)),
                  "gamma": float(.7 * rng.uniform(.9, 1.1))}
    else:
        params = {"r0": float(10000 * rng.uniform(.95, 1.05)),
                  "beta": float(3950 * rng.uniform(.97, 1.03)),
                  "phase": float(rng.uniform(-np.pi, np.pi))}
    return Sensor(kind, unit, params)


def calibration_case(sensor, budget, repeat, root_seed=20261107):
    if budget < 9:
        raise ValueError("At least nine references are required by this pilot")
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, sensor.unit, repeat, budget, 11]))
    qmin, qmax = sensor.q_range
    q = np.linspace(qmin, qmax, budget)
    q[1:-1] += rng.uniform(-.25, .25, budget - 2) * (qmax - qmin) / (budget - 1)
    adc = sensor.observe(q, rng)
    sigma = .005 if sensor.kind == "ldr" else .02
    reference = q + rng.normal(0, sigma, budget)
    tr = Transform.from_references(adc, sensor.q_range)
    val_idx = np.arange(2, budget - 1, 3)
    train_idx = np.setdiff1d(np.arange(budget), val_idx)
    # Holdout measurements use an independent RNG; increasing test size cannot
    # alter training inputs or the selected model.
    test_rng = np.random.default_rng(np.random.SeedSequence([root_seed, sensor.unit, repeat, 991]))
    q_test = np.linspace(qmin, qmax, 403)[1:-1]
    x_ideal = tr.nx(sensor.adc(q_test))
    x_noisy = tr.nx(sensor.observe(q_test, test_rng))
    x, y = tr.nx(adc), tr.ny(reference)
    payload = np.column_stack([adc, reference]).astype("<f8").tobytes()
    return dict(sensor=sensor, transform=tr, x=x, y=y, train_idx=train_idx,
                val_idx=val_idx, offset_index=int(np.argmin(abs(q - np.mean(sensor.q_range)))),
                x_ideal=x_ideal, x_noisy=x_noisy, y_test=tr.ny(q_test),
                reference_sha256=hashlib.sha256(payload).hexdigest())


@dataclass
class Model:
    family: str
    coefficients: np.ndarray
    transform: Transform
    sensor_kind: str
    hyper: dict = field(default_factory=dict)
    parent: object = None

    def predict(self, x):
        x = np.asarray(x, float)
        c = self.coefficients
        if self.family == "lut":
            return linear_basis(x, len(c)) @ c
        if self.family in ("cubic", "kan_edge"):
            return cubic_basis(x, self.hyper["n_coeff"], self.family == "kan_edge") @ c
        if self.family == "poly":
            return np.polynomial.chebyshev.chebval(x, c)
        if self.family == "residual_lut":
            return self.parent.predict(x) + linear_basis(x, len(c)) @ c
        adc = np.clip(self.transform.dx(x), 1e-7, 1 - 1e-7)
        lr = np.log(10000 * adc / (1 - adc))
        if self.family == "physics":
            if self.sensor_kind == "ldr":
                q = c[0] + c[1] * lr
            else:
                q = 1 / (c[0] + c[1] * lr + c[2] * lr ** 3) - 273.15
            return self.transform.ny(q)
        if self.family in ("factory", "offset"):
            q = make_sensor(self.sensor_kind, 0).factory(adc)
            return self.transform.ny(q) + (float(c[0]) if len(c) else 0)
        raise ValueError(self.family)

    def payload_bytes(self):
        # The common normalization metadata is counted once for a composite.
        own = self.coefficients.nbytes + COMMON_BYTES
        if self.family == "residual_lut":
            own += self.parent.payload_bytes() - COMMON_BYTES
        if self.family == "physics":
            own += 4  # divider resistance float32
        if self.family in ("factory", "offset"):
            own += 8  # two nominal inverse-model constants; R25/R_fixed = 1
        return int(own)


def fit_physics(x, y, transform, kind):
    adc = np.clip(transform.dx(x), 1e-7, 1 - 1e-7)
    lr = np.log(10000 * adc / (1 - adc))
    q = transform.dy(y)
    if kind == "ldr":
        a, target = np.column_stack([np.ones(len(x)), lr]), q
    else:
        a, target = np.column_stack([np.ones(len(x)), lr, lr ** 3]), 1 / (q + 273.15)
    coef = np.linalg.lstsq(a, target, rcond=None)[0]
    return Model("physics", coef.astype(np.float32), transform, kind)


def fit_selected(family, x_train, y_train, x_val, y_val, transform, kind, n_coeff=32):
    """Selection API deliberately has no test arguments. Refit on pooled refs."""
    x_all, y_all = np.r_[x_train, x_val], np.r_[y_train, y_val]
    best = None
    for candidate in (DEGREES if family == "poly" else LAMBDAS):
        if family == "poly":
            if candidate + 1 >= len(x_train):
                continue
            a = np.polynomial.chebyshev.chebvander(x_train, candidate)
            av = np.polynomial.chebyshev.chebvander(x_val, candidate)
            c = regularized_fit(a, y_train, 0)
        else:
            basis = linear_basis if family == "lut" else lambda z, n: cubic_basis(z, n, family == "kan_edge")
            a, av = basis(x_train, n_coeff), basis(x_val, n_coeff)
            c = regularized_fit(a, y_train, candidate, n_coeff)
        # Selection uses the same float32 coefficient precision as reporting.
        score = float(np.mean((av @ c.astype(np.float32) - y_val) ** 2))
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        raise ValueError("No feasible candidate")
    score, selected = best
    if family == "poly":
        a = np.polynomial.chebyshev.chebvander(x_all, selected)
        c = regularized_fit(a, y_all, 0)
        hyper = dict(degree=int(selected), val_mse=score)
    else:
        a = (linear_basis(x_all, n_coeff) if family == "lut" else
             cubic_basis(x_all, n_coeff, family == "kan_edge"))
        c = regularized_fit(a, y_all, selected, n_coeff)
        hyper = dict(n_coeff=n_coeff, lambda_2=float(selected), val_mse=score)
    return Model(family, c.astype(np.float32), transform, kind, hyper)


def fit_residual(x_train, y_train, x_val, y_val, transform, kind, n_coeff):
    prior = fit_physics(x_train, y_train, transform, kind)
    residual = fit_selected("lut", x_train, y_train - prior.predict(x_train),
                            x_val, y_val - prior.predict(x_val), transform, kind, n_coeff)
    parent = fit_physics(np.r_[x_train, x_val], np.r_[y_train, y_val], transform, kind)
    x, y = np.r_[x_train, x_val], np.r_[y_train, y_val]
    coef = regularized_fit(linear_basis(x, n_coeff), y - parent.predict(x),
                           residual.hyper["lambda_2"])
    return Model("residual_lut", coef.astype(np.float32), transform, kind,
                 residual.hyper, parent)


def compile_lut(model, length):
    """Sample at the exact endpoint-inclusive grid used for interpolation."""
    grid = np.linspace(-1, 1, length)
    return Model("lut", model.predict(grid).astype(np.float32), model.transform,
                 model.sensor_kind, dict(n_coeff=length, source=model.family,
                                         source_hyper=model.hyper))


@dataclass
class EncodedLUT:
    values: np.ndarray
    scale: np.float32
    offset: np.float32
    transform: Transform
    encoding: str
    hyper: dict = field(default_factory=dict)

    @classmethod
    def from_model(cls, model, encoding):
        if model.family != "lut":
            raise ValueError("Only standalone LUT models can be encoded")
        c = np.asarray(model.coefficients, np.float32)
        if not np.all(np.isfinite(c)):
            raise ValueError("Non-finite LUT values")
        if encoding == "float32":
            return cls(c.copy(), np.float32(1), np.float32(0), model.transform, encoding, model.hyper.copy())
        dtype = {"uint8": np.uint8, "int16": np.int16}[encoding]
        limits = np.iinfo(dtype)
        span = float(c.max()) - float(c.min())
        if not span:
            return cls(np.zeros(len(c), dtype=dtype), np.float32(1), c[0],
                       model.transform, encoding, model.hyper.copy())
        scale = np.float32(span / (limits.max - limits.min))
        offset = np.float32(float(c.min()) - limits.min * float(scale))
        codes = np.clip(np.rint((c.astype(float) - offset) / scale), limits.min, limits.max).astype(dtype)
        return cls(codes, scale, offset, model.transform, encoding, model.hyper.copy())

    def decoded(self):
        if self.encoding == "float32":
            return self.values
        return self.values.astype(np.float32) * self.scale + self.offset

    def predict(self, x):
        return linear_basis(x, len(self.values)) @ self.decoded()

    def payload_bytes(self):
        return int(self.values.nbytes + COMMON_BYTES + (0 if self.encoding == "float32" else 8))

    def to_dict(self):
        return dict(encoding=self.encoding, values=self.values.tolist(),
                    scale=float(self.scale), offset=float(self.offset),
                    transform=vars(self.transform), payload_bytes=self.payload_bytes(),
                    grid="endpoint_inclusive", domain=[-1, 1], hyper=self.hyper)


def error_metrics(pred, truth, y_half):
    e = (np.asarray(pred) - truth) * y_half
    if not np.all(np.isfinite(e)):
        raise ValueError("Non-finite prediction")
    return dict(mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                max_abs=float(np.max(abs(e))), bias=float(np.mean(e)))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
