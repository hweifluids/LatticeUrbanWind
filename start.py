#!/usr/bin/env python3

import subprocess
import sys
import pathlib
import threading
import datetime
import traceback

print("│───────────────────────────────────────────────────────────────────│")
print("│  Project:   WRFcpLBM - toolbox for WRF-FluidX3D coupling          │")
print("│  Module:    MAIN MODULE (Currently for preprocess)                │")
print("│  Author:    Huanxia Wei                                           │")
print("│  Email:     huanxia.wei@u.nus.edu                                 │")
print("│  Version:   <20250714A-CPU>                                       │")
print("│  License:   Currently designed for CMA, opensource in the future. │")
print("│───────────────────────────────────────────────────────────────────│")


# ──── LOGGER ─────────────────────────────────────────────────
class Logger:
    BUFFER_SIZE = 1  # Flush after every line
    def __init__(self):
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.log_file = open(
            f'{ts}.log',
            'a',
            buffering=self.BUFFER_SIZE,
            encoding=sys.getdefaultencoding(),
        )
        self.terminal = sys.__stdout__
        self._linebuf: list[str] = []      
        self.last_tqdm: str | None = None  # record last tqdm message

        # encoding
        self.encoding = self.terminal.encoding

    #  -------- write  -------- 
    def write(self, msg: str):
        if msg == '\n' and self.last_tqdm:
            cleaned = self.last_tqdm.lstrip('\r')
            if not cleaned.endswith('\n'):
                cleaned += '\n'                       # backspace the last line
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            out = f'[{ts}] {cleaned}'
            # \r backspace the last line
            self.terminal.write('\r' + out)
            self.log_file.write(out)
            self.last_tqdm = None
            return

        # -------- process tqdm messages --------
        if msg.startswith('\r'):
            # end tqdm：\rxxx\n or \r\n
            if msg.endswith('\n'):
                cleaned = msg.lstrip('\r')
                if cleaned.strip() == '':
                    cleaned = (self.last_tqdm or '').lstrip('\r')
                    if cleaned.strip() == '':
                        return  # ok fine...
                if not cleaned.endswith('\n'):
                    cleaned += '\n'  
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                out = f'[{ts}] {cleaned}'
                self.terminal.write(out)  # continue to terminal
                self.log_file.write(out)
                self.last_tqdm = None
                return
            # Refreshing progress bar: \rxxx
            if msg == self.last_tqdm:
                return
            self.last_tqdm = msg
            self.terminal.write(msg)      # write to console only
            return
        # generic message handling
        self._linebuf.append(msg)
        if msg.endswith('\n'):
            line = ''.join(self._linebuf)
            self._linebuf.clear()

            # only blank
            if line.strip() == '':
                self.terminal.write('\n')
                self.log_file.write('\n')
                return

            # add timecode to normal line
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            out = f'[{ts}] {line}'
            self.terminal.write(out)
            self.log_file.write(out)


    # -------- flush std -------- #
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        # make it uninteractive
        return False

    # -------- catch error -------- #
    def log_exception(self, exc_type, exc_value, exc_tb):
        timestamp = (datetime.datetime.now()
                     .strftime('%Y-%m-%d %H:%M:%S.') +
                     f'{int(datetime.datetime.now().microsecond / 1000):03}')
        error_message = (f'[{timestamp}] EXCEPTION: '
                         f'{"".join(traceback.format_exception(exc_type, exc_value, exc_tb))}')
        self.terminal.write(error_message)
        self.log_file.write(error_message)
        
# -------- replace stdout / stderr -------- #
sys.stdout = Logger()
sys.stderr = sys.stdout


















# === 1) Basic setup =============================================================
BRIDGE_DIR = pathlib.Path(__file__).resolve().parent / "bridge"   # bridge folder
SCRIPTS = [
    "tool_cdfInspect.py",
    "tool_shpInspect.py",
    "buildBC.py",
    "voxelization.py",
]
LOGFILE = pathlib.Path("./logfile")

# --- read conf.txt to get caseName ----------------------------------------
CONF_FILE = pathlib.Path(__file__).resolve().parent / "conf.txt"
try:
    first = CONF_FILE.read_text(encoding="utf-8").splitlines()[0]          # line 1
    key, val = [s.strip() for s in first.split("=", 1)]
    if key.lower() == "casename":
        caseName = val                                                    # conf gives
    else:
        raise ValueError
except Exception:                                                          
    caseName = input("Please input the case name: ")                                   # call interaction


# === 2) transfer STDIO =========================================================
def _forward_output(proc, log):
    """
    transfer STDOUT
    """
    for ch in iter(lambda: proc.stdout.read(1), ''):   
        print(ch, end='', flush=True)
        log.write(ch)
        log.flush()

def run_script(script: pathlib.Path, log):
    """
    transfer STDIN

    """
    # -u means unbuffered
    args = [sys.executable, "-u", str(script)]
    if script.name in ("tool_cdfInspect.py", "tool_shpInspect.py"):        # who need casename
        args.append(caseName)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,               # universal_newlines=True（Py3.7+）
        bufsize=0                # raw buffering
        # stdin=None
    )

    # main threading
    t = threading.Thread(target=_forward_output, args=(proc, log), daemon=True)
    t.start()
    proc.wait()      # allow interaction
    t.join()
    return proc.returncode

# === 3) main ============================================================
def main():

    if not BRIDGE_DIR.is_dir():
        sys.stderr.write(f"[ERROR] {BRIDGE_DIR} does not exist.\n")
        sys.exit(1)

    with LOGFILE.open("w", encoding="utf-8") as log:
        for name in SCRIPTS:
            script_path = BRIDGE_DIR / name
            if not script_path.is_file():
                msg = f"[WARN] {script_path} not found — skipping.\n"
                sys.stderr.write(msg); log.write(msg)
                continue

            print() 
            print(f"──────────────────────────────────────────────────────────────────────────────────── Running {name} ")   # Logger 会自动加时间码

            rc = run_script(script_path, log)

            print(f"──────────────────────────────────────────────────────────────────────────────────── {name} exited with code {rc} ")
            footer = f"\n--- {name} exited with code {rc} ---\n"
            log.write(footer)

            if rc != 0:
                sys.stderr.write(f"[ERROR] {name} failed (exit {rc}).\n")
        print() 
        print(f"──────────────────────────────────────────────────────────────────────────────────── All task finished as expected! ")
        print() 

if __name__ == "__main__":
    main()
