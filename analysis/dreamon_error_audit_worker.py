"""CPU forensics worker. Run ONLY through the bwrap launcher in the audit driver."""
import ast
import ctypes
import json
import os
import resource
import shutil
import sys
import time
import traceback


def snapshot(value):
    # Never invoke candidate-controlled repr, iteration, or serialization hooks.
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if type(value) in (list, tuple):
        return [snapshot(x) for x in value[:100]]
    if type(value) is dict:
        return {str(k): snapshot(v) for k, v in list(value.items())[:100]
                if type(k) in (str, int)}
    return {"unserialized_type": type(value).__name__}


def restrict():
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (4, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # default allow; namespace mounts are read-only
    for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                 "execve", "execveat", "fork", "vfork", "clone", "clone3", "ptrace",
                 "mount", "umount2", "unshare", "setns", "reboot", "bpf", "kill",
                 "tgkill", "tkill", "process_vm_writev", "open_by_handle_at", "keyctl"):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and lib.seccomp_rule_add(ctx, 0x50000 | 1, number, 0) != 0:
            raise RuntimeError("seccomp rule failed: " + name)
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp load failed")


def main():
    payload = json.load(open("/input/payload.json"))
    evaluator = open("/input/execution.py").read()
    marker = "#                     exec(check_program, exec_globals)"
    assert evaluator.count(marker) == 1
    ns = {"__name__": "pinned_evaluator_enabled"}
    exec(compile(evaluator.replace(marker, "                    exec(check_program, exec_globals)"),
                 "/input/execution.py", "exec"), ns)
    restrict()
    if payload.get("probe"):
        import socket
        denied = {}
        for name, action in (("network", lambda: socket.socket()),
                             ("shell", lambda: os.system("true")),
                             ("host_write", lambda: open("/usr/audit_probe", "w")),
                             ("credential", lambda: open("/home/shx/.codex/auth.json"))):
            try:
                value = action()
                denied[name] = value != 0 if name == "shell" else False
            except OSError as exc:
                denied[name] = True
        print(json.dumps({"probe": denied, "environment": dict(os.environ)}))
        return
    r = payload["problem"]
    started = time.monotonic()
    cpu = time.process_time()
    result = []
    calls = []
    if payload["mode"] == "official":
        ns["unsafe_execute"](r, payload["output"], result, 3.0)
        info = {"audit_passed": result == ["passed"], "audit_failure_reason": result[0]}
    else:
        program = r["prompt"] + payload["output"] + r["suffix"] + "\n" + r["test"]
        globals_ = {}
        cleanup = shutil.rmtree, os.rmdir, os.chdir
        with ns["create_tempdir"]():
            ns["reliability_guard"]()
            try:
                with ns["swallow_io"](), ns["time_limit"](3.0):
                    exec(compile(program, "candidate.py", "exec"), globals_)
                    original = globals_[r["entry_point"]]
                    def proxy(*args, **kwargs):
                        before = snapshot(args)
                        kw = snapshot(kwargs)
                        try:
                            value = original(*args, **kwargs)
                        except BaseException as exc:
                            calls.append({"input": before, "kwargs": kw, "exception_type": type(exc).__name__, "exception": str(exc)})
                            raise
                        calls.append({"input": before, "kwargs": kw, "actual": snapshot(value)})
                        return value
                    globals_["check"](proxy)
                info = {"audit_passed": True, "audit_failure_reason": "passed"}
            except BaseException as exc:
                tb = traceback.extract_tb(exc.__traceback__)
                frame = exc.__traceback__
                locals_ = {}
                while frame:
                    if frame.tb_frame.f_code.co_filename == "candidate.py":
                        locals_ = {k: snapshot(v) for k, v in frame.tb_frame.f_locals.items() if not k.startswith("__")}
                    frame = frame.tb_next
                info = {"audit_passed": False, "audit_failure_reason": "timed out" if type(exc).__name__ == "TimeoutException" else "failed: " + str(exc),
                        "exception_type": type(exc).__name__, "traceback": [dict(file=x.filename, line=x.lineno, function=x.name) for x in tb],
                        "failure_locals": locals_}
                relevant = [x for x in tb if x.filename == "candidate.py"]
                if relevant:
                    line = relevant[-1].lineno
                    info["failure_line"] = line
                    info["failure_source"] = program.splitlines()[line-1]
            finally:
                shutil.rmtree, os.rmdir, os.chdir = cleanup
        info["calls"] = calls[-5:]
        info["call_count"] = len(calls)
        info["expected"] = "见原断言；性质/关系断言不强制转换为单一expected"
    info.update(cpu_sec=time.process_time()-cpu, wall_sec=time.monotonic()-started,
                timeout_origin="SIGALRM wall timer or original exception; see mode", mode=payload["mode"])
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
