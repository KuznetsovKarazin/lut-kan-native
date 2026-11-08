"""Bounded UART greeting recovery; mocked serial values are not MCU evidence."""
from collections import deque
import io
import pytest

from mcu.capture import capture_stream
from mcu.capture_matched import parse_lines, MEGA_HELLO_DELAY_S, MEGA_HELLO_ERROR_BUDGET
from test_matched_capture import fixture


class GreetingDevice:
    def __init__(self, reject=1, fail_phase=None, response="ERROR,INVALID_COMMAND"):
        self.t=0.0;self.reject=reject;self.fail_phase=fail_phase;self.response=response
        self.pending=deque();self.commands=[];self.sent_at=[];self.hellos=0
    def write(self, data):
        self.commands.append(data);self.sent_at.append(self.t)
        if data==b"H\n":
            self.hellos+=1
            line=self.response if self.hellos<=self.reject else "READY,SENSOR_MATCHED_V1,mega2560"
            self.pending.append((line+"\r\n").encode())
        elif data.startswith(b"Q,"):
            token=data.decode().strip()[2:]
            line="ERROR,INVALID_COMMAND" if self.fail_phase=="echo" else "ECHO,"+token
            self.pending.append((line+"\r\n").encode())
        elif data.startswith(b"R,"):
            token=data.decode().strip()[2:]
            if self.fail_phase=="accept":lines=["ERROR,INVALID_COMMAND"]
            else:
                lines=fixture(token=token)[2:]
                if self.fail_phase=="running":lines.insert(12,"ERROR,INVALID_COMMAND")
            self.pending.append(("\r\n".join(lines)+"\r\n").encode())
        return len(data)
    def read(self, count):
        self.t+=.1
        return self.pending.popleft() if self.pending else b""


def run(device, raw, session, budget=MEGA_HELLO_ERROR_BUDGET):
    return capture_stream(device,raw,True,"a"*32,40,session,clock=lambda:device.t,emit=lambda _:None,
                          ready_line="READY,SENSOR_MATCHED_V1,mega2560",
                          hello_delay_s=MEGA_HELLO_DELAY_S,hello_error_budget=budget)


@pytest.mark.parametrize("reject",[0,1,3])
def test_only_greeting_retries_and_original_bytes_remain(reject):
    device=GreetingDevice(reject);raw=io.BytesIO();session={"tx_commands":[]}
    result=parse_lines(run(device,raw,session))
    assert device.sent_at[0]>=2.0
    assert device.commands.count(b"H\n")==reject+1
    assert sum(c.startswith(b"Q,") for c in device.commands)==1
    assert sum(c.startswith(b"R,") for c in device.commands)==1
    assert len(session["pre_ready_rejections"])==reject
    assert raw.getvalue().count(b"ERROR,INVALID_COMMAND\r\n")==reject
    assert len(result["update_timings"])==120 and session["usb_phase"]=="complete"


def test_repeated_bad_greeting_aborts_without_requesting_a_run():
    device=GreetingDevice(99);raw=io.BytesIO();session={"tx_commands":[]}
    with pytest.raises(ValueError,match="budget exceeded"):run(device,raw,session)
    assert device.commands==[b"H\n"]*4
    assert len(session["pre_ready_rejections"])==3
    assert raw.getvalue().count(b"ERROR,INVALID_COMMAND")==4


@pytest.mark.parametrize("phase",["echo","accept","running"])
def test_command_errors_after_ready_are_fatal_and_run_is_never_retried(phase):
    device=GreetingDevice(0,fail_phase=phase)
    with pytest.raises(ValueError,match="Device rejected command"):
        run(device,io.BytesIO(),{"tx_commands":[]})
    assert sum(c.startswith(b"R,") for c in device.commands)==(0 if phase=="echo" else 1)


@pytest.mark.parametrize("budget,response",[(0,"ERROR,INVALID_COMMAND"),(3,"ERROR,OTHER")])
def test_recovery_is_not_enabled_for_other_policies_or_errors(budget,response):
    device=GreetingDevice(response=response)
    with pytest.raises(ValueError,match="Device rejected command"):
        run(device,io.BytesIO(),{"tx_commands":[]},budget)
    assert device.commands==[b"H\n"]


@pytest.mark.parametrize("case",["late","excess","other","c3"])
def test_saved_log_parser_rejects_invalid_recovery_prefix(case):
    lines=fixture(board="esp32c3" if case=="c3" else "mega2560")
    error="ERROR,OTHER" if case=="other" else "ERROR,INVALID_COMMAND"
    if case=="late":lines.insert(1,error)
    else:lines=[error]*(4 if case=="excess" else 1)+lines
    with pytest.raises(ValueError,match="pre-READY"):parse_lines(lines)
