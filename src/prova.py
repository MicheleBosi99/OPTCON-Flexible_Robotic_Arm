def main():
    # Initial state and input
    X = np.array([0.1, 0.1, 0.1, 0.1])
    U = np.array([1, 0, 0, 0])
    
    # Substitute numerical values into M
    M_num = M.subs({m1: 2, m2: 2, l1: 1.5, l2: 1.5, r1: 0.75, r2: 0.75, I1: 1.5, I2: 1.5, g: 9.81, f1: 0.1, f2: 0.1})
    
    # Perform the inversion with numerical values
    A_upper = -M_num.inv() @ F
    A_lower = sp.eye(2)
    A = sp.BlockMatrix([[A_upper, sp.zeros(2, 2)], [A_lower, sp.zeros(2, 2)]]).as_explicit()
    
    print("A:", A)

if __name__ == "__main__":
    main()