#!/usr/bin/env python3

import subprocess
import sys
import pathlib
import threading
import datetime
import traceback
import os
import locale
import codecs

print("|-------------------------------------------------------------------|")
print("|  Project:   WRFcpLBM - toolbox for WRF-FluidX3D coupling          |")
print("|  Module:    MAKELUW - THE MAKING OF LUW                           |")
print("|  Author:    Huanxia Wei                                           |")
print("|  Email:     huanxia.wei@u.nus.edu                                 |")
print("|  Version:   <20251031A-GPU>                                       |")
print("|  License:   Customized License.                                   |")
print("|-------------------------------------------------------------------|")


# ──── LOGGER ─────────────────────────────────────────────────
class Logger:
    BUFFER_SIZE = 1  # Flush after every line
    def __init__(self):
# 日志路径：$ProjectFolder/proj_temp/$yyyymmddhhmmss$.log，失败则回落到 $pid$.log
        try:
            ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        except Exception:
            ts = None
        deck_path = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path('.').resolve()
        project_folder = deck_path.parent
        log_dir = project_folder / 'proj_temp'
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        log_name = f'{ts}.log' if ts else f'{os.getpid()}.log'
        self.log_file = open(
            str(log_dir / log_name),
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
LUW_HOME = pathlib.Path(os.environ["LUW_HOME"])
SCRIPTS = [
    LUW_HOME / "core/tools_core/cdfInspect.py",
    LUW_HOME / "core/tools_core/shpInspect.py",
    LUW_HOME / "core/bridge_core/1_buildBC.py",
    LUW_HOME / "core/bridge_core/2_shpCutter.py",
    LUW_HOME / "core/bridge_core/3_voxelization.py",
    LUW_HOME / "core/tools_core/prerunValidate.py",
]

# --- read deck path from command line ----------------------------------------
if len(sys.argv) != 2:
    sys.stderr.write("Usage: makeluw.py <deck_file_path>\n")
    sys.exit(1)
deck_path = pathlib.Path(sys.argv[1]).resolve()

# === 2) transfer STDIO =========================================================
def _forward_output(proc, terminal, log):
    """
    Forward child STDOUT/STDERR to the user's terminal and to the log file.

    Notes:
    - Child processes write prompts without trailing newlines (e.g. input("...: ")).
      We must forward partial lines immediately; do not rely on line-buffering.
    - Avoid routing through Logger/print(), which buffers until newline.
    """
    if proc.stdout is None:
        return

    # IMPORTANT:
    # Do NOT use TextIOWrapper.read(N) here; on pipes it may block until N chars are read,
    # which makes console output appear "stuck" for a long time. Use os.read() to stream
    # whatever bytes are currently available.
    encoding = locale.getpreferredencoding(False) or "utf-8"
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")

    fd = proc.stdout.fileno()
    while True:
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        text = decoder.decode(data, final=False)
        if text:
            terminal.write(text)
            terminal.flush()
            log.write(text)
            log.flush()

    # Flush any remaining decoder state (in case the stream ends mid-codepoint)
    tail = decoder.decode(b"", final=True)
    if tail:
        terminal.write(tail)
        terminal.flush()
        log.write(tail)
        log.flush()

def run_script(script: pathlib.Path, log):
    """
    transfer STDIN

    """
    # -u means unbuffered
    args = [sys.executable, "-u", str(script), str(deck_path)]

    # If this launcher is executed in a pipeline / redirected context, sys.stdin may not be
    # connected to the interactive console, causing input() in child scripts to hit EOFError.
    # In that case, try to explicitly bind child stdin to the console device.
    child_stdin = None
    child_stdin_needs_close = False
    try:
        stdin_src = sys.__stdin__ if sys.__stdin__ is not None else sys.stdin
        if stdin_src is not None:
            try:
                if stdin_src.isatty():
                    # Explicitly pass stdin to avoid Windows handle inheritance edge-cases when
                    # stdout is redirected to a pipe.
                    child_stdin = stdin_src
                else:
                    # Prefer real console input even if we are running in a pipeline.
                    if os.name == "nt":
                        child_stdin = open("CONIN$", "r", encoding=sys.getdefaultencoding(), errors="replace")
                    else:
                        child_stdin = open("/dev/tty", "r", encoding=sys.getdefaultencoding(), errors="replace")
                    child_stdin_needs_close = True
            except Exception:
                child_stdin = None  # fallback: inherit whatever stdin we have

        proc = subprocess.Popen(
            args,
            stdin=child_stdin,         # may be None to inherit
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,                # stream bytes; decode in _forward_output()
            bufsize=0,                 # unbuffered pipe reader
        )
    finally:
        if child_stdin_needs_close and child_stdin is not None:
            try:
                child_stdin.close()
            except Exception:
                pass

    # main threading
    terminal = getattr(sys.stdout, "terminal", sys.__stdout__)
    t = threading.Thread(target=_forward_output, args=(proc, terminal, log), daemon=True)
    t.start()
    proc.wait()      # allow interaction
    t.join()
    return proc.returncode

# === 3) main ============================================================
def main():

    with pathlib.Path(sys.stdout.log_file.name).open("a", encoding="utf-8") as log:
        for script_path in SCRIPTS:
            if not script_path.is_file():
                msg = f"[WARN] {script_path} not found, skipping.\n"
                sys.stderr.write(msg); log.write(msg)
                continue

            print() 
            print(f"──────────────────────────────────────────────────────────────────────────────────── Running {script_path.name} ")

            rc = run_script(script_path, log)

            print(f"──────────────────────────────────────────────────────────────────────────────────── {script_path.name} exited with code {rc} ")
            footer = f"\n--- {script_path.name} exited with code {rc} ---\n"
            log.write(footer)

            if rc != 0:
                sys.stderr.write(f"[ERROR] {script_path.name} failed (exit {rc}).\n")
        print() 
        print(f"──────────────────────────────────────────────────────────────────────────────────── All task finished as expected! ")
        print() 

if __name__ == "__main__":
    main()
