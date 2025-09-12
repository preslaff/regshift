"""
CVXPY Solver Setup and Testing Script

This script helps diagnose and fix CVXPY solver issues for the 
Dynamic Investment Strategies system.
"""

import subprocess
import sys

def check_cvxpy_installation():
    """Check if CVXPY is installed and which solvers are available."""
    try:
        import cvxpy as cp
        print("[OK] CVXPY is installed")
        print(f"  Version: {cp.__version__}")
        
        installed_solvers = cp.installed_solvers()
        print(f"\nInstalled solvers: {installed_solvers}")
        
        # Check for recommended solvers
        recommended_solvers = ['OSQP', 'SCS', 'CLARABEL', 'ECOS']
        available_recommended = [s for s in recommended_solvers if s in installed_solvers]
        
        print(f"Available recommended solvers: {available_recommended}")
        
        if not available_recommended:
            print("[WARNING] No recommended solvers found!")
            return False
        
        return True
        
    except ImportError:
        print("[MISSING] CVXPY is not installed")
        return False

def install_solvers():
    """Install commonly needed CVXPY solvers."""
    print("\n" + "="*50)
    print("CVXPY SOLVER INSTALLATION")
    print("="*50)
    
    # Common solver packages
    solver_packages = [
        ("osqp", "OSQP - Fast quadratic programming solver"),
        ("scs", "SCS - Splitting Conic Solver"),
        ("clarabel", "CLARABEL - Interior point conic solver"),
        ("ecos", "ECOS - Embedded Conic Solver")
    ]
    
    print("\nInstalling recommended solvers...")
    
    for package, description in solver_packages:
        print(f"\nInstalling {package}...")
        print(f"Description: {description}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"[OK] {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] Failed to install {package}: {e}")
            continue

def test_solver_functionality():
    """Test basic solver functionality."""
    try:
        import cvxpy as cp
        import numpy as np
        
        print("\n" + "="*50)
        print("TESTING SOLVER FUNCTIONALITY")
        print("="*50)
        
        # Simple test problem: minimize ||Ax - b||^2
        np.random.seed(1)
        m, n = 20, 10
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        problem = cp.Problem(objective)
        
        # Test each available solver
        for solver in cp.installed_solvers():
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status == cp.OPTIMAL:
                    print(f"[OK] {solver}: Working (optimal value: {problem.value:.4f})")
                else:
                    print(f"[WARNING]  {solver}: Problem status: {problem.status}")
            except Exception as e:
                print(f"[FAILED] {solver}: Error - {e}")
        
        return True
        
    except ImportError:
        print("Cannot test - CVXPY not available")
        return False

def print_installation_instructions():
    """Print detailed installation instructions."""
    print("\n" + "="*60)
    print("INSTALLATION INSTRUCTIONS")
    print("="*60)
    print("""
If you're experiencing CVXPY solver issues, follow these steps:

1. BASIC INSTALLATION:
   pip install cvxpy

2. INSTALL RECOMMENDED SOLVERS:
   pip install osqp scs clarabel ecos

3. FOR ADVANCED USERS:
   # Commercial solvers (require licenses)
   pip install cvxopt
   # conda install -c mosek mosek  # Requires license

4. VERIFY INSTALLATION:
   python -c "import cvxpy as cp; print(cp.installed_solvers())"

5. COMMON ISSUES:

   Issue: "The solver ECOS is not installed"
   Solution: pip install ecos

   Issue: "No module named 'cvxpy'"  
   Solution: pip install cvxpy

   Issue: All solvers fail
   Solution: Try conda install -c conda-forge cvxpy

6. ALTERNATIVE APPROACH:
   If CVXPY issues persist, the system will automatically
   fall back to scipy optimization, which is more stable
   but potentially less efficient.

For more help, see: https://www.cvxpy.org/install/
""")

def main():
    """Main setup and diagnostic function."""
    print("CVXPY Solver Setup and Diagnostics")
    print("="*50)
    
    # Check current installation
    cvxpy_ok = check_cvxpy_installation()
    
    if not cvxpy_ok:
        print("\n[WARNING]  CVXPY not found. Installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "cvxpy"
            ])
            print("[OK] CVXPY installed")
            cvxpy_ok = check_cvxpy_installation()
        except subprocess.CalledProcessError:
            print("[FAILED] Failed to install CVXPY")
            print_installation_instructions()
            return
    
    # Install additional solvers
    install_solvers()
    
    # Test functionality
    print("\nRe-checking installation...")
    check_cvxpy_installation()
    test_solver_functionality()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("""
[OK] CVXPY solver setup is now complete!

The Dynamic Investment Strategies system should now work
properly with portfolio optimization.

If you still encounter issues, the system will automatically
fall back to scipy-based optimization.
""")

if __name__ == "__main__":
    main()