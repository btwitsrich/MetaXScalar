import subprocess
import time
import sys

print("Starting server...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,  # capture stderr separately
    text=True,
)

time.sleep(4)

print("Starting inference...")
res = subprocess.run([sys.executable, "inference.py"], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("ERRORS:")
    print(res.stderr)

proc.terminate()
try:
    outs, errs = proc.communicate(timeout=5)
except Exception:
    proc.kill()
    outs, errs = proc.communicate()

print("\n--- SERVER STDOUT ---")
print(outs or "(no output)")
if errs:
    print("\n--- SERVER STDERR ---")
    print(errs)
