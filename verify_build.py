
import srvdb
import sys

print(f"Python: {sys.version}")
print(f"SrvDB Location: {srvdb.__file__}")
print(f"SrvDB Doc: {srvdb.__doc__}")

print("\nChecking capabilities:")
try:
    print(f"new_product_quantized exists: {hasattr(srvdb.SrvDBPython, 'new_product_quantized')}")
except Exception as e:
    print(f"Error checking new_product_quantized: {e}")

try:
    db = srvdb.SrvDBPython("test_ivf_check", 128, mode="ivf")
    print("IVF mode init: PASS")
except Exception as e:
    print(f"IVF mode init: FAIL ({e})")
