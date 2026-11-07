"""Validate device-log rejection paths with artificial in-memory protocol records.

These records are test inputs only, never hardware measurements or result files.
"""
import pytest
from mcu.capture import parse_lines

def protocol_fixture():
    methods=["overhead","physics","polynomial","cubic16","kan16","uniform32_float32","uniform32_int16","uniform32_uint8","nonuniform16_int16"]
    lines=["SENSOR_BENCH_V1,TEST_INPUT_ONLY","BOARD,mega2560","CPU_HZ,16000000","FREE_MEMORY_ESTIMATE,1000"]
    for name in methods:
        lines.append(f"PAYLOAD,{name},88")
        if name!="overhead":lines.append(f"ACCURACY,{name},0.1,0.2,0.3,0.00001,PASS")
        lines.extend(f"TIME,{name},{r},2048,4096,12.0" for r in range(10))
    lines.extend(f"UPDATE_COST_ONLY,{r},2048,8192,MONOTONE" for r in range(10))
    return lines+["END_SENSOR_BENCH"]

def test_complete_capture_and_summary():
    result=parse_lines(protocol_fixture())
    assert result["summary"]["physics"]["gross_us_per_call"]==2
    assert result["summary"]["physics"]["overhead_subtracted_us_per_call"]==0
    assert len(result["accuracy"])==8

@pytest.mark.parametrize("corruption",["missing_end","duplicate_repeat","nonfinite","bad_parity","wrong_clock","reset"])
def test_reject_invalid_capture(corruption):
    lines=protocol_fixture()
    if corruption=="missing_end":lines.pop()
    elif corruption=="duplicate_repeat":lines.insert(-1,"TIME,physics,0,2048,4096,12.0")
    elif corruption=="nonfinite":lines=[x.replace("0.00001,PASS","nan,PASS") for x in lines]
    elif corruption=="bad_parity":lines=[x.replace("0.00001,PASS","0.1,PASS") for x in lines]
    elif corruption=="wrong_clock":lines[2]="CPU_HZ,8000000"
    elif corruption=="reset":lines.insert(50,"SENSOR_BENCH_V1,TEST_INPUT_ONLY")
    with pytest.raises(ValueError):parse_lines(lines)
