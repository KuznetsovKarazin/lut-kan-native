"""Transport fault tests with artificial streams, never hardware result files."""
import hashlib
import io
import json
import subprocess
from collections import deque
from pathlib import Path

import pytest
from mcu import capture
from test_capture import protocol_fixture


class MockDevice:
    def __init__(self, terminate=True):
        self.pending=deque();self.commands=[];self.t=0.;self.terminate=terminate
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def write(self,data):
        self.commands.append(data)
        if data==b"H\n":self.pending.extend([b"READ",b"Y,SENSOR_USB_V3,esp32c3\r",b"\n"])
        elif data.startswith(b"Q,"):
            self.pending.append(b"ECHO,"+data[2:].strip()+b"\r\n")
        elif data.startswith(b"R,"):
            token=data.decode().strip().split(",")[1]
            lines=protocol_fixture()
            lines[1:3]=["BOARD,esp32c3","CPU_HZ,160000000"]
            lines[1:1]=["SESSION,"+token,"TRANSPORT,usb-jtag-poll-command-v3"]
            lines.insert(0,"ACCEPTED,"+token)
            if not self.terminate:lines.pop()
            raw=("\r\n".join(lines)+"\r\n").encode()
            self.pending.extend(raw[i:i+7] for i in range(0,len(raw),7))
        return len(data)
    def read(self,n):
        if self.pending:return self.pending.popleft()
        self.t+=.1
        return b""


def test_split_lines_and_single_start_command():
    stream=MockDevice();raw=io.BytesIO();token="a"*32;session={"tx_commands":[]}
    lines=capture.capture_stream(stream,raw,True,token,3,session,emit=lambda s:None,clock=lambda:stream.t)
    result=capture.parse_lines(lines)
    assert stream.commands==[b"H\n",b"Q,"+token.encode()+b"\n",b"R,"+token.encode()+b"\n"]
    assert result["metadata"]["SESSION"]==token
    assert b"END_SENSOR_BENCH\r\n" in raw.getvalue()


def test_missing_end_preserves_partial_output():
    stream=MockDevice(terminate=False);raw=io.BytesIO()
    with pytest.raises(TimeoutError):
        capture.capture_stream(stream,raw,True,"a"*32,3,{"tx_commands":[]},emit=lambda s:None,clock=lambda:stream.t)
    assert b"TIME,polynomial," in raw.getvalue()


def test_control_lines_set_before_open():
    class Port:
        def __init__(self,**kwargs):
            assert kwargs["port"] is None
            assert kwargs["write_timeout"]==2
        def open(self):assert self.dtr is False and self.rts is False and self.port=="COM_TEST"
    capture.open_serial("COM_TEST",True,serial_factory=Port)


def test_two_command_sessions_have_distinct_ids_and_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,"open_serial",lambda *args:MockDevice())
    ids=[]
    for n in range(2):
        out=tmp_path/f"mock_{n}.json"
        monkeypatch.setattr("sys.argv",["capture.py","--port","MOCK_ONLY","--out",str(out),"--start-command"])
        capture.main()
        session=json.loads(out.with_suffix(".session.json").read_text())
        result=json.loads(out.read_text())
        assert session["status"]=="validated"
        assert session["usb_phase"]=="complete" and session["run_accepted_utc"]
        assert session["raw_sha256"]==hashlib.sha256(out.with_suffix(".log").read_bytes()).hexdigest()
        assert result["metadata"]["SESSION"]==session["session_id"]
        ids.append(session["session_id"])
    assert ids[0]!=ids[1]


def test_disconnect_saves_failure_metadata(tmp_path,monkeypatch):
    class BrokenDevice(MockDevice):
        def read(self,n):raise OSError("Mock USB disconnect")
    monkeypatch.setattr(capture,"open_serial",lambda *args:BrokenDevice())
    out=tmp_path/"mock_failure.json"
    monkeypatch.setattr("sys.argv",["capture.py","--port","MOCK_ONLY","--out",str(out),"--start-command"])
    with pytest.raises(SystemExit) as err:capture.main()
    assert err.value.code==1 and not out.exists()
    session=json.loads(out.with_suffix(".session.json").read_text())
    assert session["status"]=="failed" and "disconnect" in session["error"]
    assert out.with_suffix(".log").exists()


def test_bounded_firmware_command_parser(tmp_path):
    include=Path(__file__).resolve().parent/"mcu/include"
    cpp=tmp_path/"command_test.cpp"
    cpp.write_text(r'''
#include "command_protocol.h"
#include <cassert>
#include <string>
int main(){
 BenchCommand p;
 const std::string token("d2d81a1695da4b559ea86a493bd379ae");
 for(char c:std::string("H\r\n"))if(p.push(c))assert(p.hello());
 int starts=0;
 for(char c:std::string("R,")+token+"\nH\nR,"+token+"\n")
   if(p.push(c) && p.session())++starts;
 assert(starts==2);
 for(char c:std::string("Q,")+token+"\n")if(p.push(c)){
   assert(p.probe() && token==p.probe());assert(!p.session());
 }
 for(char c:std::string(100,'x')+"\n")assert(!p.push(c));
 for(char c:std::string("R,not-a-token\n"))if(p.push(c))assert(!p.session());
 for(char c:std::string("H\n"))if(p.push(c))assert(p.hello());
}
''')
    exe=tmp_path/"commands"
    subprocess.run(["g++","-std=c++11","-I",str(include),str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


@pytest.mark.parametrize("stop,phase,message",[
    ("H","echo","no full-length command echo"),
    ("Q","accept","no ACCEPTED response"),
    ("R","header","no benchmark header")])
def test_startup_failure_identifies_phase_without_retry(stop,phase,message):
    class StoppedDevice(MockDevice):
        def write(self,data):
            if data[:1].decode() in {"H":"H","Q":"HQ","R":"HQR"}[stop]:
                if data.startswith(b"R,"):
                    self.commands.append(data)
                    self.pending.append(b"ACCEPTED,"+data[2:].strip()+b"\n")
                    return len(data)
                return super().write(data)
            self.commands.append(data);return len(data)
    stream=StoppedDevice();session={"tx_commands":[]}
    with pytest.raises(TimeoutError,match=message):
        capture.capture_stream(stream,io.BytesIO(),True,"a"*32,40,session,
                               emit=lambda s:None,clock=lambda:stream.t)
    assert session["usb_phase"]==phase
    runs=[x for x in stream.commands if x.startswith(b"R,")]
    assert len(runs)==(0 if stop=="H" else 1)


def test_wrong_echo_does_not_start_benchmark():
    class WrongEcho(MockDevice):
        def write(self,data):
            if data.startswith(b"Q,"):
                self.commands.append(data);self.pending.append(b"ECHO,"+b"b"*32+b"\n");return len(data)
            return super().write(data)
    stream=WrongEcho()
    with pytest.raises(ValueError,match="mismatched USB command echo"):
        capture.capture_stream(stream,io.BytesIO(),True,"a"*32,40,{"tx_commands":[]},
                               emit=lambda s:None,clock=lambda:stream.t)
    assert not any(x.startswith(b"R,") for x in stream.commands)


def test_old_firmware_rejected_before_run():
    class OldFirmware(MockDevice):
        def write(self,data):
            self.commands.append(data);self.pending.append(b"READY,SENSOR_USB_V2,esp32c3\r\n");return len(data)
    stream=OldFirmware()
    with pytest.raises(ValueError,match="current C3 v3 build"):
        capture.capture_stream(stream,io.BytesIO(),True,"a"*32,40,{"tx_commands":[]},
                               emit=lambda s:None,clock=lambda:stream.t)
    assert stream.commands==[b"H\n"]
