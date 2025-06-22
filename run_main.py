import subprocess
import sys
import os

# The script we want to execute
target_script = "examples/customize/build_graph/pipeline/kg_builder_from_code.py"

# The python interpreter from the correct environment
python_executable = r"D:\anaconda3\envs\graphrag\python.exe"

print(f"Attempting to run '{target_script}' with interpreter '{python_executable}'...")

# Set the PYTHONPATH to include the current directory
env = os.environ.copy()
env["PYTHONPATH"] = "."

try:
    # Execute the script using subprocess, which correctly handles __name__ == "__main__"
    result = subprocess.run(
        [python_executable, target_script],
        capture_output=True,
        text=True,
        check=True,  # This will raise an exception if the script returns a non-zero exit code
        env=env,
        encoding='utf-8' # Ensure correct encoding for output
    )

    print("\n--- Script Output ---")
    print(result.stdout)
    if result.stderr:
        print("\n--- Script Errors ---")
        print(result.stderr)

    print("\n✅ Script executed successfully.")

except subprocess.CalledProcessError as e:
    print(f"\n❌ ERROR: The script '{target_script}' failed with exit code {e.returncode}.")
    print("\n--- Script Output (stdout) ---")
    print(e.stdout)
    print("\n--- Script Errors (stderr) ---")
    print(e.stderr)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find the python executable at '{python_executable}'. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
