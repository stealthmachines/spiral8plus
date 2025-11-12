import os
import sys
import subprocess

# Simulate what the control panel does
env = os.environ.copy()
env['GENOME_LIMIT'] = '10000'
env['GENOME_CHROMOSOME'] = 'NC_000001.11'

# Test if the subprocess receives the environment variables
test_code = """
import os
print("Environment variables received:")
print(f"  GENOME_LIMIT = {os.environ.get('GENOME_LIMIT', 'NOT SET')}")
print(f"  GENOME_CHROMOSOME = {os.environ.get('GENOME_CHROMOSOME', 'NOT SET')}")
"""

result = subprocess.run(
    [sys.executable, '-c', test_code],
    env=env,
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
print(f"Return code: {result.returncode}")
