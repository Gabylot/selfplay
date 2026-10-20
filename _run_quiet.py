"""Run a diag script and print all output (incl. tracebacks) to stdout,
so PowerShell's stderr mangling can't hide anything."""
import runpy
import sys
import traceback

script = sys.argv[1]
sys.argv = sys.argv[1:]  # so the target script's argparse sees its own args

try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
