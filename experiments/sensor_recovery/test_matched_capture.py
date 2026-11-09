"""Reject incomplete/mixed captures and exercise the actual firmware's output."""
import io,json,subprocess
from pathlib import Path
import pytest
from mcu.capture_matched import parse_lines,STATIC,ONLINE,MODES,CHECKS
from mcu.run_matched import capture_runs,sha
from mcu.capture import capture_stream
from test_capture_transport import MockDevice

HERE=Path(__file__).resolve().parent


def fixture(board="mega2560",token="a"*32):
    manifest=json.loads((HERE/"mcu/matched_manifest.json").read_text())
    lines=["READY,SENSOR_MATCHED_V1,"+board,"ECHO,"+token,"ACCEPTED,"+token,
        "SENSOR_BENCH_V1,synthetic_ntc_frontend,seed2026110801,unit0,B75","SUITE,matched_adaptation_v1",
        "BOARD,"+board,"SESSION,"+token,"TRANSPORT,"+("uart-command-matched-v1" if board=="mega2560" else "usb-jtag-poll-command-matched-v1"),
        "CPU_HZ,"+("16000000" if board=="mega2560" else "160000000"),"PRECISION,float32","FIXTURE_SHA256,"+manifest["fixture_sha256"]]
    for m in STATIC:lines += [f"PAYLOAD,{m},144",f"ACCURACY,{m},0.05,0.06,0.1,0.00001,PASS"]
    for m in ("overhead",*STATIC):lines += [f"TIME,{m},{r},2048,4096,123" for r in range(10)]
    for m in ONLINE:lines.append(f"UPDATE_STORAGE,{m},148,"+str(160 if "projected" in m else 0))
    for d in MODES:
        lines += [f"UPDATE_OVERHEAD,{d},{r},384,768,123" for r in range(10)]
        for m in ONLINE:
            order="MONOTONE" if "projected" in m else "UNCONSTRAINED"
            lines += [f"STATE,{d},{m},{t},0.00001,{order},PASS" for t in CHECKS]
            lines.append(f"UPDATE_ACCURACY,{d},{m},0.05,0.06,0.1,0.00001,PASS")
            lines += [f"UPDATE_TIME,{d},{m},{r},384,768,123,0.00001,PASS" for r in range(10)]
    return lines+["END_SENSOR_BENCH"]


@pytest.mark.parametrize("board",["mega2560","esp32c3"])
def test_complete_synthetic_protocol_fixture(board):
    result=parse_lines(fixture(board));assert len(result["state_checks"])==96 and len(result["update_timings"])==120


@pytest.mark.parametrize("change",["missing","duplicate","nan","wrong_payload","wrong_fixture","bad_order","reset","no_ack","wrong_clock","wrong_mode"])
def test_reject_modified_protocol(change):
    lines=fixture()
    if change=="missing":lines=[s for s in lines if not s.startswith("STATE,step,lut_nlms,128,")]
    elif change=="duplicate":lines.insert(-1,next(s for s in lines if s.startswith("UPDATE_TIME,")))
    elif change=="nan":lines=[s.replace("0.00001","nan") if s.startswith("STATE,") else s for s in lines]
    elif change=="wrong_payload":lines=[s.replace(",144",",145") if s.startswith("PAYLOAD,") else s for s in lines]
    elif change=="wrong_fixture":lines=["FIXTURE_SHA256,"+"0"*64 if s.startswith("FIXTURE_SHA256,") else s for s in lines]
    elif change=="bad_order":lines=[s.replace("MONOTONE","FAIL") for s in lines]
    elif change=="reset":lines.insert(-1,lines[3])
    elif change=="no_ack":lines=[s for s in lines if not s.startswith("ACCEPTED,")]
    elif change=="wrong_clock":lines=[s.replace("CPU_HZ,16000000","CPU_HZ,8000000") for s in lines]
    elif change=="wrong_mode":lines=[s.replace("UPDATE_TIME,step,","UPDATE_TIME,other,") for s in lines]
    with pytest.raises(ValueError):parse_lines(lines)


@pytest.mark.parametrize("board",["mega2560","esp32c3"])
def test_shared_transport_accepts_only_requested_matched_handshake(board):
    class Device(MockDevice):
        def write(self,data):
            if data==b"H\n":
                self.commands.append(data)
                raw=("READY,SENSOR_MATCHED_V1,"+board+"\r\n").encode()
                self.pending.extend(raw[i:i+5] for i in range(0,len(raw),5))
                return len(data)
            return super().write(data)
    d=Device();session={"tx_commands":[]}
    capture_stream(d,io.BytesIO(),True,"a"*32,40,session,clock=lambda:d.t,emit=lambda s:None,
                   ready_line="READY,SENSOR_MATCHED_V1,"+board)
    assert session["usb_phase"]=="complete"


def test_two_capture_reparse_and_device_session_matching(tmp_path):
    metadata={"captures":[]};count=0
    def runner(command,console):
        nonlocal count
        count+=1;target=Path(command[command.index("--out")+1]);token=str(count)*32
        lines=fixture(token=token);raw=target.with_suffix(".log");raw.write_text("\n".join(lines)+"\n")
        target.write_text(json.dumps(parse_lines(lines)))
        side=dict(status="validated",session_id=token,raw_sha256=sha(raw),result_sha256=sha(target))
        target.with_suffix(".session.json").write_text(json.dumps(side))
    capture_runs("mega2560","TEST_ONLY",tmp_path,2,metadata,runner)
    assert len(metadata["captures"])==2 and metadata["captures"][0]["session_id"]!=metadata["captures"][1]["session_id"]


def test_real_firmware_serialization_with_mock_clock(tmp_path):
    # This executes target C++ arithmetic/output on a host. The artificial clock
    # below is ONLY a protocol test and must never be saved as MCU timing evidence.
    (tmp_path/"avr").mkdir()
    (tmp_path/"avr/pgmspace.h").write_text('#pragma once\n#include <cstring>\n#define PROGMEM\ninline float pgm_read_float(const float *p){return *p;}\ninline void *memcpy_P(void *a,const void *b,size_t n){return std::memcpy(a,b,n);}\n')
    (tmp_path/"Arduino.h").write_text('''#pragma once
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <string>
struct FakeSerial {
 std::string data;size_t at=0;void begin(int){}int available(){return at<data.size();}
 int read(){return available()?data[at++]:-1;}
 template<class T> void print(T v){std::cout<<v;}
 void print(float v,int n){std::cout<<std::fixed<<std::setprecision(n)<<v;}
 template<class T> void println(T v){print(v);std::cout<<"\\n";}
 void println(float v,int n){print(v,n);std::cout<<"\\n";}
};
extern FakeSerial Serial;
inline uint32_t micros(){static uint32_t t=0;return t+=100;}
inline void delay(int){}
''')
    source=tmp_path/"main.cpp"
    source.write_text('#include "Arduino.h"\nFakeSerial Serial;\n#include "'+str(HERE/"mcu/src/matched_main.cpp")+'"\nint main(){setup();Serial.data="H\\nQ,'+'a'*32+'\\nR,'+'a'*32+'\\n";loop();}\n')
    exe=tmp_path/"firmware"
    subprocess.run(["g++","-O2","-std=c++11","-ffp-contract=off","-D__AVR__","-DF_CPU=16000000UL",
        "-DSENSOR_MATCHED_SUITE","-I",str(tmp_path),"-I",str(HERE/"mcu/include"),str(source),"-o",str(exe)],check=True)
    r=subprocess.run([str(exe)],capture_output=True,text=True,check=True)
    result=parse_lines(r.stdout.splitlines())
    assert len(result["accuracy"])==4 and len(result["update_accuracy"])==12
    assert max(v["max_gap_C"] for v in result["state_checks"])<.002
