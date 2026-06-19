import subprocess, sys
result = subprocess.run(
    [r"f:\python\selfplay\chess_rust\build_with_msvc.bat"],
    capture_output=True, text=True, shell=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
sys.exit(result.returncode)