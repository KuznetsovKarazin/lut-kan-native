"""Verify saved campaign integrity and replay a fixed sample without refitting settings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from calibration import make_sensor, calibration_case, canonical_hash, error_metrics
from matched_adaptation import (static_models, model_record, serialized_payload,
    reference_record, initial_models, generate_stream, stream_record, trajectory)


def records(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    root = a.results
    here = Path(__file__).resolve().parent
    infos = {}
    for stage in ("static", "development", "confirmation"):
        info = json.loads((root / stage / "run_info.json").read_text())
        assert info["complete"], stage
        for name, expected in info["source_sha256"].items():
            assert hashlib.sha256((here / name).read_bytes()).hexdigest() == expected, name
        for name, expected in info["result_sha256"].items():
            assert hashlib.sha256((root / stage / name).read_bytes()).hexdigest() == expected, name
        infos[stage] = info
    assert infos["development"]["seed"] != infos["confirmation"]["seed"]
    selected = root / "development" / "selected_parameters.json"
    assert selected.read_bytes() == (root / "confirmation" / selected.name).read_bytes()
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == infos["confirmation"]["selection_sha256"]

    counts = {}; replay_models = replay_trajectories = 0
    samples = {}
    for r in records(root / "static" / "cases.jsonl.gz"):
        if r["unit"] == 0 and r["repeat"] == 0:
            case = calibration_case(make_sensor(r["scenario"], 0, infos["static"]["seed"]),
                                    r["budget"], 0, infos["static"]["seed"])
            assert all(r[k] == v for k, v in reference_record(case).items())
            samples[r["case_id"]] = (case, {(cap, name): m for cap, name, m in static_models(case)})
    seen = set()
    for r in records(root / "static" / "raw_results.jsonl.gz"):
        key = (r["case_id"], r["cap_bytes"], r["method"])
        assert key not in seen
        seen.add(key)
        if r["case_id"] not in samples:
            continue
        case, models = samples[r["case_id"]]
        model = models[(r["cap_bytes"], r["method"])]
        assert model_record(model) == r["model"]
        assert hashlib.sha256(serialized_payload(model)).hexdigest() == r["payload_sha256"]
        for label, x in (("ideal", "x_ideal"), ("noisy", "x_noisy")):
            for k, v in error_metrics(model.predict(case[x]), case["y_test"], case["transform"].y_half).items():
                assert r[label + "_" + k] == v
        replay_models += 1
    counts["static_rows"] = len(seen)
    assert len(seen) == infos["static"]["result_rows"]

    replay_streams = {}
    for stage in ("development", "confirmation"):
        stream_hashes = {}; seed = infos[stage]["seed"]
        for r in records(root / stage / "streams.jsonl.gz"):
            expected = r.pop("stream_sha256")
            assert canonical_hash(r) == expected
            assert r["case_id"] not in stream_hashes
            stream_hashes[r["case_id"]] = expected
            if stage == "confirmation" and r["unit"] == 0 and r["repeat"] == 0:
                case = calibration_case(make_sensor(r["scenario"], 0, seed), r["budget"], 0, seed)
                models = initial_models(case)
                stream = generate_stream(case, 0, r["budget"], r["drift"], seed)
                assert reference_record(case) == r["initial_references"]
                assert {k: model_record(v) for k, v in models.items()} == r["initial_models"]
                assert stream_record(stream) == r["stream"]
                replay_streams[r["case_id"]] = (models, stream)
        assert len(stream_hashes) == infos[stage]["streams"]
        counts[stage + "_streams"] = len(stream_hashes)
        raw = "candidate_results.jsonl.gz" if stage == "development" else "raw_trajectories.jsonl.gz"
        seen = set()
        for r in records(root / stage / raw):
            key = (r["case_id"], r["method"], r["parameter"])
            assert key not in seen
            seen.add(key)
            assert stream_hashes[r["case_id"]] == r["stream_sha256"]
            if stage == "confirmation":
                assert r["failure"] is None and r["completed_steps"] == 384
                if r["case_id"] in replay_streams:
                    models, stream = replay_streams[r["case_id"]]
                    result = trajectory(models, stream, r["method"], r["parameter"], retain=True)
                    assert all(r[k] == v for k, v in result.items()), key
                    replay_trajectories += 1
        assert len(seen) == infos[stage]["trajectories"]
        counts[stage + "_trajectories"] = len(seen)
    result = dict(status="pass", **counts, replay_selection="unit=0, repeat=0, every scenario/budget/drift/method",
        exact_static_model_replays=replay_models, exact_confirmation_trajectory_replays=replay_trajectories,
        exact_confirmation_stream_replays=len(replay_streams),
        verified="source/result SHA256, stream links, unique inventories, selection identity, exact numerical replay")
    text = json.dumps(result, indent=2) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
