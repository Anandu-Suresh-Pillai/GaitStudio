import numpy as np
from ik.analytical_ik import LegIK

def run_verification():
    print("==================================================")
    print("        GAITSTUDIO KINEMATICS VERIFICATION        ")
    print("==================================================")
    
    # Instantiate solvers for left and right legs
    ik_left = LegIK(leg_name="FL", y_sign=1.0)
    ik_right = LegIK(leg_name="FR", y_sign=-1.0)
    
    # Test cases: (leg_solver, description, q1, q2, q3)
    test_cases = [
        (ik_left, "Left Leg - Default Stance", 0.0, 0.0, -1.5),
        (ik_left, "Left Leg - Hip Abduction", 0.4, 0.5, -2.0),
        (ik_left, "Left Leg - High Flexion", -0.4, 1.2, -1.0),
        (ik_right, "Right Leg - Default Stance", 0.0, 0.0, -1.5),
        (ik_right, "Right Leg - Hip Adduction", -0.3, -0.2, -1.8),
        (ik_right, "Right Leg - Full Extension", 0.2, 0.8, -0.3),
    ]
    
    success = True
    for solver, desc, q1, q2, q3 in test_cases:
        # 1. Forward Kinematics
        pos = solver.solve_fk(q1, q2, q3)
        
        # 2. Inverse Kinematics
        q_sol = solver.solve_ik(pos)
        
        if q_sol is None:
            print(f"[-] {desc}: FAIL (IK returned None)")
            success = False
            continue
            
        # 3. Forward Kinematics on IK solution
        pos_rec = solver.solve_fk(*q_sol)
        
        # Error metrics
        pos_err = np.linalg.norm(pos - pos_rec)
        q_err = np.linalg.norm(np.array([q1, q2, q3]) - np.array(q_sol))
        
        # Note: multiple joint configurations can sometimes result in the same foot position (e.g., knee forward vs backward), 
        # but our solver enforces negative q3 (knee backward), which should match our active stances.
        if pos_err < 1e-4:
            print(f"[+] {desc}: SUCCESS")
            print(f"    Orig Joint: [{q1:.3f}, {q2:.3f}, {q3:.3f}]")
            print(f"    Solved Joint: [{q_sol[0]:.3f}, {q_sol[1]:.3f}, {q_sol[2]:.3f}]")
            print(f"    Foot Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"    Cartesian Error: {pos_err:.2e} m")
        else:
            print(f"[-] {desc}: FAIL (Reconstruction Error: {pos_err:.2e} m)")
            success = False
            
    # Run a randomized sweep in realistic locomotion joint ranges
    print("\n--- Running 1000 Randomized Pose Sweeps (Locomotion Range) ---")
    num_tests = 1000
    errors = []
    
    for _ in range(num_tests):
        solver = ik_left if np.random.rand() > 0.5 else ik_right
        # Generate random joint values within realistic stance/swing locomotion limits
        q1 = np.random.uniform(-0.4, 0.4)
        q2 = np.random.uniform(-0.4, 0.8)
        q3 = np.random.uniform(-2.0, -0.8)
        
        pos = solver.solve_fk(q1, q2, q3)
        q_sol = solver.solve_ik(pos)
        pos_rec = solver.solve_fk(*q_sol)
        
        err = np.linalg.norm(pos - pos_rec)
        errors.append(err)
        
    max_err = np.max(errors)
    mean_err = np.mean(errors)
    print(f"Sweep Completed!")
    print(f"Mean Reconstruction Error: {mean_err:.2e} m")
    print(f"Maximum Reconstruction Error: {max_err:.2e} m")
    
    if max_err < 1e-4:
        print("[+] Kinematics engine verified as ROBUST and 100% ACCURATE!")
    else:
        print("[-] Verification failed under certain poses.")
        success = False
        
    print("==================================================")
    return success

if __name__ == "__main__":
    run_verification()
