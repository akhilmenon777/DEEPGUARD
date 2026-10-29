"""Test if backend can be imported"""
import sys
import traceback

try:
    print("Attempting to import backend.main...")
    from backend.main import app
    print("✓ SUCCESS: App imported successfully!")
    print(f"✓ App has {len(app.routes)} routes")
except Exception as e:
    print("✗ ERROR importing backend.main:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
