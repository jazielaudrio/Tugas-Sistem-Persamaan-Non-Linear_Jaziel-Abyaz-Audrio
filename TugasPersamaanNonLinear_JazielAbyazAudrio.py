# Nama: Jaziel Abyaz Audrio
# NIM: 21120123130107
# Tugas Metode Numerik: Penyelesaian Sistem Persamaan Non-Linear

import pandas as pd
import numpy as np

def g1(x, y):
    val = 10 - x * y
    return np.sqrt(val) if val >= 0 else float('nan')

def g2(x, y):
    return 57 - 3 * x * y**2

def metode_seidel(x0, y0, epsilon, max_iter=10):
    x, y = x0, y0
    history = [[0, x, y, 0, 0]]
    
    print("\n### 2. Metode Iterasi Seidel (g1A, g2B) ###")
    
    for i in range(1, max_iter):
        x_old, y_old = x, y
        
        x_new = g1(x, y)
        
        if np.isnan(x_new):
            print("\nSolusi Divergen! (x menjadi NaN)")
            break
            
        y_new = g2(x_new, y)
        
        if np.isnan(y_new) or abs(y_new - y) > 1e10:
            print("\nSolusi Divergen!")
            break
        
        x, y = x_new, y_new
        deltaX = abs(x - x_old)
        deltaY = abs(y - y_old)
        history.append([i, x, y, deltaX, deltaY])
        
        if deltaX < epsilon and deltaY < epsilon:
            print("\nSolusi Konvergen ditemukan.")
            break

    df = pd.DataFrame(history, columns=['Iterasi', 'x', 'y', 'deltaX', 'deltaY'])
    print(df.to_string(index=False))

def u(x, y): return x**2 + x*y - 10
def v(x, y): return y + 3*x*y**2 - 57
def du_dx(x, y): return 2*x + y
def du_dy(x, y): return x
def dv_dx(x, y): return 3*y**2
def dv_dy(x, y): return 1 + 6*x*y

def metode_newton_raphson(x0, y0, epsilon, max_iter=50):
    x, y = x0, y0
    history = [[0, x, y, 0, 0]]
    
    print("\n### 3. Metode Newton-Raphson ###")

    for i in range(1, max_iter):
        x_old, y_old = x, y

        u_val, v_val = u(x, y), v(x, y)
        du_dx_val, du_dy_val = du_dx(x, y), du_dy(x, y)
        dv_dx_val, dv_dy_val = dv_dx(x, y), dv_dy(x, y)

        J = du_dx_val * dv_dy_val - du_dy_val * dv_dx_val
        if abs(J) < 1e-12:
            print("\nDeterminan Jacobian mendekati nol. Solusi gagal.")
            break

        x_new = x - (u_val * dv_dy_val - v_val * du_dy_val) / J
        y_new = y - (v_val * du_dx_val - u_val * dv_dx_val) / J
        
        x, y = x_new, y_new
        deltaX = abs(x - x_old)
        deltaY = abs(y - y_old)
        history.append([i, x, y, deltaX, deltaY])
        
        if deltaX < epsilon and deltaY < epsilon:
            print("\nSolusi Konvergen ditemukan.")
            break
            
    df = pd.DataFrame(history, columns=['Iterasi', 'x', 'y', 'deltaX', 'deltaY'])
    print(df.to_string(index=False))

if __name__ == "__main__":
    x_awal, y_awal = 1.5, 3.5
    toleransi = 0.000001
    
    metode_seidel(x_awal, y_awal, toleransi)

    metode_newton_raphson(x_awal, y_awal, toleransi)