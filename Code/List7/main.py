from RungeKutta import RungeKutta

def main()->None:
    ft = lambda t, y: 1 + (t - y) ** 2
        
    rk = RungeKutta(ft, 2, 3, 0.1, 1)
    
    # 3/8 rule 4th order Runge-Kutta method
    rk.SetButcherTableau(method = "ThreeEights")
    rk.RungeKuttaMethod()

    print(rk.sol)

if __name__ == "__main__":
    main()