"""Run the frozen sensor recovery pilot and retain every result and reference."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from calibration import (Model, EncodedLUT, make_sensor, calibration_case,
    fit_selected, fit_physics, fit_residual, compile_lut, error_metrics,
    canonical_hash, linear_basis, regularized_fit, adam_fit, objective)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_case(kind, unit, repeat, budget, lengths, root_seed, optimizer):
    sensor = make_sensor(kind, unit, root_seed)
    case = calibration_case(sensor, budget, repeat, root_seed)
    x, y, tr = case["x"], case["y"], case["transform"]
    it, iv = case["train_idx"], case["val_idx"]
    args = x[it], y[it], x[iv], y[iv], tr, kind
    rows = []
    models = {}

    def add(name, model, length=0, encoding="float32", refs=budget):
        ideal = error_metrics(model.predict(case["x_ideal"]), case["y_test"], tr.y_half)
        noisy = error_metrics(model.predict(case["x_noisy"]), case["y_test"], tr.y_half)
        row = dict(scenario=kind, unit=unit, repeat=repeat, budget=budget,
                   used_references=refs, n_train=len(it), n_val=len(iv),
                   method=name, length=length, encoding=encoding,
                   payload_bytes=model.payload_bytes(), physical_unit=sensor.quantity,
                   reference_sha256=case["reference_sha256"],
                   test_clipped_fraction=float(np.mean(abs(case["x_noisy"]) > 1)),
                   **{"ideal_" + k: v for k, v in ideal.items()},
                   **{"noisy_" + k: v for k, v in noisy.items()},
                   hyper=getattr(model, "hyper", {}))
        rows.append(row)

    def add_lut(name, model, length):
        for fmt in ("float32", "uint8", "int16"):
            enc = EncodedLUT.from_model(model, fmt)
            add(name, enc, length, fmt)
            if unit == repeat == 0:
                models[f"{name}_L{length}_{fmt}"] = enc.to_dict()

    factory = Model("factory", np.empty(0, np.float32), tr, kind)
    add("factory", factory, refs=0)
    ix = case["offset_index"]
    delta = np.float32(y[ix] - factory.predict(x[ix:ix+1])[0])
    add("onepoint_offset", Model("offset", np.array([delta], np.float32), tr, kind), refs=1)
    add("polynomial", fit_selected("poly", *args))
    physics = fit_physics(x, y, tr, kind)
    add("physics", physics)
    for length in lengths:
        lut = fit_selected("lut", *args, n_coeff=length)
        add_lut("direct_lut", lut, length)
        add_lut("physics_to_lut", compile_lut(physics, length), length)
        cubic = fit_selected("cubic", *args, n_coeff=length)
        add("cubic_spline", cubic, length)
        kan = fit_selected("kan_edge", *args, n_coeff=length)
        add("kan_edge", kan, length)
        add_lut("kan_to_lut", compile_lut(kan, length), length)
        add("physics_residual_lut", fit_residual(*args, n_coeff=length), length)

    diagnostic = None
    if optimizer:
        # Same objective, same data, same representation. No validation/test
        # selection: this is an optimization diagnostic, not an extra contender.
        a = linear_basis(x, 32)
        ls = regularized_fit(a, y, 1.0)
        ad = adam_fit(a, y, 1.0)
        diagnostic = dict(scenario=kind, unit=unit, repeat=repeat, budget=budget,
                          objective_ls=objective(a, y, ls, 1.0),
                          objective_adam=objective(a, y, ad, 1.0),
                          coefficient_max_gap=float(np.max(abs(ls-ad))))
    reference = dict(scenario=kind, unit=unit, repeat=repeat, budget=budget,
                     parameters=sensor.parameters, transform=vars(tr),
                     x=x.tolist(), y=y.tolist(), train_idx=it.tolist(), val_idx=iv.tolist(),
                     offset_index=ix, reference_sha256=case["reference_sha256"],
                     x_ideal=case["x_ideal"].tolist(), x_noisy=case["x_noisy"].tolist(),
                     y_test=case["y_test"].tolist())
    return rows, reference, models, diagnostic


def aggregate(rows):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in ("scenario", "budget", "method", "length", "encoding"))
        groups.setdefault(key, []).append(r)
    result = []
    for key, values in sorted(groups.items()):
        unit_ids = sorted({v["unit"] for v in values})
        per_unit = [np.mean([v["noisy_mae"] for v in values if v["unit"] == u]) for u in unit_ids]
        rng = np.random.default_rng(712)
        draws = rng.choice(per_unit, (2000, len(per_unit)), replace=True).mean(axis=1)
        row = dict(zip(("scenario", "budget", "method", "length", "encoding"), key))
        payloads = [v["payload_bytes"] for v in values]
        row.update(payload_bytes_min=min(payloads), payload_bytes_max=max(payloads),
                   mean_payload_bytes=float(np.mean(payloads)))
        row.update(n_units=len(unit_ids), n_cases=len(values),
                   mean_noisy_mae=float(np.mean(per_unit)),
                   unit_bootstrap_mae_lo=float(np.quantile(draws, .025)),
                   unit_bootstrap_mae_hi=float(np.quantile(draws, .975)),
                   mean_ideal_mae=float(np.mean([v["ideal_mae"] for v in values])),
                   worst_noisy_max_abs=float(max(v["noisy_max_abs"] for v in values)),
                   mean_noisy_rmse=float(np.mean([v["noisy_rmse"] for v in values])),
                   mean_bias=float(np.mean([v["noisy_bias"] for v in values])),
                   unit_mean_noisy_mae={str(u): float(m) for u, m in zip(unit_ids, per_unit)})
        result.append(row)
    return result


def paired_comparisons(rows):
    """Paired unit bootstrap of LUT-vs-compiled edge at identical payload."""
    out = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for budget in sorted({r["budget"] for r in rows}):
            for length in sorted({r["length"] for r in rows if r["method"] == "direct_lut"}):
                for encoding in ("float32", "uint8", "int16"):
                    subset = [r for r in rows if r["scenario"] == scenario and r["budget"] == budget
                              and r["length"] == length and r["encoding"] == encoding]
                    by_method = {m: {(r["unit"],r["repeat"]):r for r in subset if r["method"] == m}
                                 for m in ("direct_lut", "physics_to_lut", "kan_to_lut")}
                    for other in ("physics_to_lut", "kan_to_lut"):
                        ids = sorted(set(by_method["direct_lut"]) & set(by_method[other]))
                        units = sorted({i[0] for i in ids})
                        if not ids:
                            continue
                        a, b = [], []
                        for u in units:
                            ui = [i for i in ids if i[0] == u]
                            a.append(np.mean([by_method["direct_lut"][i]["noisy_mae"] for i in ui]))
                            b.append(np.mean([by_method[other][i]["noisy_mae"] for i in ui]))
                        a, b = np.asarray(a), np.asarray(b)
                        ix = np.random.default_rng(914).integers(0, len(units), (2000, len(units)))
                        diff = (a[ix] - b[ix]).mean(axis=1)
                        out.append(dict(scenario=scenario,budget=budget,length=length,encoding=encoding,
                            comparator=other,n_units=len(units),
                            mean_difference_direct_minus_other=float(np.mean(a-b)),
                            bootstrap_difference_lo=float(np.quantile(diff,.025)),
                            bootstrap_difference_hi=float(np.quantile(diff,.975)),
                            ratio_other_over_direct=float(np.mean(b)/np.mean(a))))
    return out


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--units", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--budgets", type=int, nargs="+", default=[30,75])
    p.add_argument("--lengths", type=int, nargs="+", default=[8,16,32,64])
    p.add_argument("--scenarios", nargs="+", default=["ntc","ntc_frontend","ldr"])
    p.add_argument("--root-seed", type=int, default=20261107)
    p.add_argument("--skip-optimizer", action="store_true")
    args = p.parse_args()
    if args.units < 1 or args.repeats < 1:
        p.error("units and repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "run_info.json").exists():
        p.error("Output already contains a completed run; choose a fresh path")
    started = time.time()
    all_rows, diags = [], []
    total = len(args.scenarios)*len(args.budgets)*args.units*args.repeats
    count = 0
    with gzip.open(args.out / "cases.jsonl.gz", "wt", encoding="utf-8") as casefile:
        for scenario in args.scenarios:
            for budget in args.budgets:
                for unit in range(args.units):
                    for repeat in range(args.repeats):
                        rows, ref, models, diag = run_case(scenario,unit,repeat,budget,args.lengths,
                                                       args.root_seed,not args.skip_optimizer)
                        all_rows.extend(rows)
                        casefile.write(json.dumps(ref,allow_nan=False)+"\n")
                        if diag:
                            diags.append(diag)
                        if models:
                            dump_json(args.out / f"deployment_examples_{scenario}_B{budget}.json", models)
                        count += 1
                        if count % 8 == 0 or count == total:
                            print(f"Completed {count}/{total} cases, {time.time()-started:.1f}s", flush=True)
    with gzip.open(args.out/"raw_results.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row,allow_nan=False)+"\n")
    summary = aggregate(all_rows)
    dump_json(args.out / "summary.json", summary)
    with (args.out / "summary.csv").open("w",newline="",encoding="utf-8") as f:
        fields = [k for k in summary[0] if k != "unit_mean_noisy_mae"]
        writer=csv.DictWriter(f,fields,lineterminator="\n");writer.writeheader()
        writer.writerows([{k:v for k,v in r.items() if k in fields} for r in summary])
    dump_json(args.out/"paired_comparisons.json",paired_comparisons(all_rows))
    dump_json(args.out/"optimizer_diagnostics.json",diags)
    try:
        head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        dirty = bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    except (OSError,subprocess.SubprocessError):
        head, dirty = None, None
    sources = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(HERE.iterdir()) if path.suffix in (".py", ".c") or path.name=="PROTOCOL.md"}
    manifest = dict(provenance="synthetic_parametric_simulation_not_physical_measurements",
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        base_git_head=head,working_tree_dirty=dirty,source_content_hashes=sources,
        cases=count,result_rows=len(all_rows),elapsed_seconds=time.time()-started,
        hardware_measurements=False,primary_metric="noisy_mae",
        bootstrap="paired unit means over repeats; 2000 resamples; exploratory",
        threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")})
    dump_json(args.out/"run_info.json",manifest)
    print(json.dumps({"cases":count,"rows":len(all_rows),"seconds":round(time.time()-started,2)}))


if __name__ == "__main__":
    main()
