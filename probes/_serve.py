"""界面探针的「服务有人在跑吗」小工具（2026-10-29 新建）。

为什么要有它
------------
`probes/*.py` 都假设 8501 端口已经有人把服务跑起来了，它们只管点页面。
这个假设平时靠"上一个界面检查顺手起了一个、退出时没关"来满足 ——
一旦顺序变了（或那一项先失败），后面的探针就会报
`net::ERR_CONNECTION_REFUSED`，看起来像"界面挂了"，其实是没人开服务。

这里统一成一条规则：**能用现成的就用，没有就自己起一个，跑完自己关掉。**
自己起的才关，别人起的不动（否则会把还在跑的检查踢掉）。

用法
----
    from _serve import ensure_server
    proc = ensure_server("8501")     # 已有服务 -> 返回 None；自己起的 -> 返回 Popen
    try:
        ...
    finally:
        stop_server(proc)
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


#: 探测本机服务**必须不走代理**（2026-10-29 踩坑）。
#: 沙箱 / 公司网络 / 抓包工具会把 `HTTP_PROXY` 设进环境，`urlopen` 照着走，
#: 于是 `http://127.0.0.1:8501/` 拿到 **502 Bad Gateway** → `_alive()` 判「没服务」→
#: 探针又起一个实例、连到错的那个，最后整项报「页面没渲染」。
#: 实测：4 项界面检查全挂（一项白等 405s），手工开服务后同一份代码全过。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _alive(url: str) -> bool:
    try:
        with _OPENER.open(url, timeout=3) as r:
            return 200 <= getattr(r, "status", 200) < 400
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _pids_listening(port: int) -> list[int]:
    """监听这个端口的进程号（只认 LISTENING；TIME_WAIT 那行没有进程号）。"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:                      # noqa: BLE001
        return []
    pids: list[int] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        local, state, pid = parts[1], parts[3].upper(), parts[4]
        if not local.endswith(f":{port}") or state != "LISTENING":
            continue
        if pid.isdigit() and int(pid) > 0:
            pids.append(int(pid))
    return pids


def free_port(port: str | int) -> int:
    """杀掉端口上残留的**本机服务进程**，返回清掉几个。

    为什么必须按端口、不能按进程名（2026-10-30 踩坑）
    --------------------------------------------------
    服务是用 `python.exe -m streamlit run app.py` 起的，**进程名是 python.exe**，
    所以 `taskkill /IM streamlit.exe` 一个也杀不到 —— 残留实例一直占着端口。
    后面的界面检查连上去，拿到的却是上一轮留下的页面状态（比如已经跑完的旧结果），
    于是等不到自己要的元素，白等到超时。实测代价：全套里 4 项界面检查失败
    （最长白等 404 秒），而单独跑同一份代码全部通过。

    只杀 python 进程：85xx 段是本项目测试专用，但万一有人拿它跑了别的东西，
    也不至于误伤非 Python 的服务。
    """
    killed = 0
    for pid in _pids_listening(int(port)):
        try:
            name = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15).stdout.lower()
        except Exception:                  # noqa: BLE001
            continue
        if "python" not in name:
            continue
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=15)
            killed += 1
        except Exception:                  # noqa: BLE001
            pass
    return killed


def ensure_server(port: str = "8501", wait_s: int = 120):
    """端口上没服务就起一个。返回自己起的进程（没有则返回 None）。"""
    url = f"http://127.0.0.1:{port}/"
    if _alive(url):
        print(f"[serve] 复用已在 {port} 端口运行的服务")
        return None
    # 服务没响应、但端口上还占着进程 —— 多半是上一项留下的半死实例。
    # 不清掉就起新的会撞端口，新实例直接退出，然后这里白等到超时。
    n = free_port(port)
    if n:
        print(f"[serve] 端口 {port} 上有 {n} 个残留实例已清理")
        time.sleep(1.0)
    log = ROOT / "cache" / f"probe_server_{port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [PY, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port), "--server.headless", "true"],
        cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if _alive(url):
            print(f"[serve] 已自行启动服务（{time.monotonic() - t0:.1f}s，日志 {log}）")
            return proc
        if proc.poll() is not None:
            print(f"[serve] 服务启动失败，见 {log}")
            return proc
        time.sleep(1.5)
    print(f"[serve] 等超时（{wait_s}s），见 {log}")
    return proc


def stop_server(proc) -> None:
    """只关自己起的那个；别人起的传给 None 时什么也不做。"""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=12)
    except Exception:                       # noqa: BLE001
        try:
            proc.kill()
        except Exception:                   # noqa: BLE001
            pass


if __name__ == "__main__":                  # 手动排查用
    p = ensure_server(sys.argv[1] if len(sys.argv) > 1 else "8501")
    if p is not None:
        print("自己起的，按 Ctrl+C 结束")
