#!/usr/bin/env python3
"""
Test script to verify all imports work across the project
"""

import sys
import os

def test_imports():
    """Test imports from different directories"""
    
    # Test 1: From project root
    print("Testing from project root...")
    try:
        from emergent_simulator import initialize_grid, rule_diffusion
        print("✅ Direct import from project root: SUCCESS")
    except ImportError as e:
        print(f"❌ Direct import from project root: FAILED - {e}")
    
    # Test 2: From notebooks directory
    print("\nTesting from notebooks directory...")
    os.chdir('notebooks')
    sys.path.append('..')
    try:
        from emergent_simulator import initialize_grid, rule_diffusion
        print("✅ Import from notebooks with path fix: SUCCESS")
    except ImportError as e:
        print(f"❌ Import from notebooks with path fix: FAILED - {e}")
    
    # Test 3: From streamlit_app directory
    print("\nTesting from streamlit_app directory...")
    os.chdir('../streamlit_app')
    sys.path.append('..')
    try:
        from emergent_simulator import initialize_grid, rule_diffusion
        print("✅ Import from streamlit_app with path fix: SUCCESS")
    except ImportError as e:
        print(f"❌ Import from streamlit_app with path fix: FAILED - {e}")
    
    print("\n🎉 All import tests completed!")

if __name__ == "__main__":
    test_imports()
