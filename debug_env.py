
import sys
import os
import srvdb
import pprint

print(f"--- DEBUG ENV ---")
print(f"CWD: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"SrvDB File: {srvdb.__file__}")
print(f"SrvDB Doc: {srvdb.__doc__}")

print("\n--- SrvDB Module Structure ---")
pprint.pprint(dir(srvdb))

print("\n--- SrvDBPython Class Structure ---")
try:
    pprint.pprint(dir(srvdb.SrvDBPython))
except Exception as e:
    print(f"ERROR: Could not inspect SrvDBPython: {e}")

print("\n--- Specific Method Checks ---")
methods_to_check = ['new_product_quantized', 'new_hnsw_quantized', 'new_scalar_quantized', 'configure_ivf']
for method in methods_to_check:
    exists = hasattr(srvdb.SrvDBPython, method)
    print(f"{method}: {exists}")

if not hasattr(srvdb.SrvDBPython, 'new_product_quantized'):
    print("\nCRITICAL: 'new_product_quantized' is missing!")
    print("This means the Python binding binary is outdated or stale.")
else:
    print("\nSUCCESS: 'new_product_quantized' is present.")
