"""Exercise the Windows redirected-output failure using real child processes."""
import io
import json
import os
import sys
import zipfile

import pytest

from mcu.run_suite import run_logged
from mcu import run_matched


def test_unicode_child_and_grandchild_survive_cp1251_console(tmp_path,monkeypatch):
    # Reproduce both boundaries: redirected child stdout and the parent's console.
    monkeypatch.setenv("PYTHONIOENCODING","cp1251")
    monkeypatch.setenv("PYTHONUTF8","0")
    payload="\u251c\u2500\u2500 package \u03bc \u043f\u0430\u043a\u0435\u0442"
    raw=io.BytesIO()
    console=io.TextIOWrapper(raw,encoding="cp1251",errors="strict",write_through=True)
    monkeypatch.setattr(sys,"stdout",console)
    child="print("+ascii(payload)+")"
    script=("import subprocess,sys; "+child+"; sys.stdout.flush(); "
            "subprocess.run([sys.executable,'-c',"+repr(child)+"],check=True)")
    path=tmp_path/"packages.txt"
    run_logged([sys.executable,"-c",script],path)
    assert path.read_text(encoding="utf8")==2*(payload+"\n")
    assert "\\u251c\\u2500\\u2500 package \\u03bc" in raw.getvalue().decode("cp1251")
    assert os.environ["PYTHONIOENCODING"]=="cp1251"  # No global environment change.


def test_failed_child_keeps_unicode_log_and_failure_status(tmp_path,monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING","cp1251")
    path=tmp_path/"failed.txt"
    with pytest.raises(RuntimeError,match="Command exited 7; see failed.txt"):
        run_logged([sys.executable,"-c","import sys; print('\\u2514\\u2500 failure'); sys.exit(7)"],path)
    assert path.read_text(encoding="utf8")=="\u2514\u2500 failure\n"


def test_inventory_failure_archives_images_without_starting_capture(tmp_path,monkeypatch):
    # A failed inventory must stay a failed batch, with the correct stage.
    # All image/log bytes below are mocks, never hardware evidence.
    here=tmp_path/"mcu";build=here/".pio/build/esp32c3_matched"
    build.mkdir(parents=True)
    for name in ("firmware.elf","firmware.bin"):(build/name).write_bytes(b"MOCK_IMAGE")
    monkeypatch.setattr(run_matched,"HERE",here)
    monkeypatch.setattr(run_matched,"source_hashes",lambda:{"MOCK_SOURCE":{}})
    def runner(command,path):
        if "pkg" in command:
            path.write_text("MOCK_INVENTORY_FAILURE\n",encoding="utf8")
            raise RuntimeError("Command exited 7; see packages.txt")
        path.write_text("MOCK_UPLOAD\n",encoding="utf8")
    monkeypatch.setattr(run_matched,"run_logged",runner)
    def no_capture(*args,**kwargs):pytest.fail("Capture must not start after inventory failure")
    monkeypatch.setattr(run_matched,"wait_for_port",no_capture)
    monkeypatch.setattr(run_matched,"capture_runs",no_capture)
    out=tmp_path/"mock_batch"
    monkeypatch.setattr(sys,"argv",["run_matched.py","--board","esp32c3","--port","MOCK_ONLY","--out",str(out)])
    with pytest.raises(SystemExit,match="Batch stopped"):run_matched.main()
    with zipfile.ZipFile(out.with_suffix(".zip")) as archive:
        batch=json.loads(archive.read("batch.json"))
        assert batch["status"]=="failed" and batch["stage"]=="package_inventory"
        assert batch["captures"]==[]
        assert archive.read("firmware/firmware.bin")==b"MOCK_IMAGE"
        assert archive.read("packages.txt")==b"MOCK_INVENTORY_FAILURE\n"
