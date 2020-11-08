import numpy as np
import scipy.signal
import sympy
import mpmath
import matplotlib.pyplot as plt

# precition
mpmath.mp.prec = 256

#
def cascade_wavelet(phi, g, J):
    div = 2**J
    sqrt2 = np.sqrt(2)

    x = np.linspace(-3, 4, num=div*7, endpoint=False)
    psi = np.zeros(div*7)
    for k, gk in enumerate(g):
        for i in range(len(psi)):
            if 0 <= 2*i-div*k < len(phi):
                psi[i] += sqrt2*gk*phi[2*i-div*k]
    return x, psi
#
def dual_cascade(h, g, J=7):
    phi_x, phi_y, _ = scipy.signal.cascade(h, J=J)
    phi_x -= (len(h)-1)//2
    psi_x, psi_y = cascade_wavelet(phi_y, g, J=J)
    return phi_x, phi_y, psi_x, psi_y

def cdf_9_7():
    # 多項式を作成する q(y) = 20y^3 + 10*y^2 + 4*y + 1
    qy = [mpmath.binomial(4-1+k,k) for k in [3,2,1,0]]

    # q(y) = 0 を解く
    y = mpmath.polyroots(qy)

    # y[0]: -0.3423840948
    # y[1]: -0.0788079525 + 0.3739306454j
    # y[2]: -0.0788079525 - 0.3739306454j

    # 実数解からなる因子を使用して多項式を作成
    h0z = sympy.sympify('-sqrt(2.0)*(y-y0)/y0') \
               .subs({'y':'-1/4*z+1/2-1/4/z', 'y0':y[0]})

    # vanising momentsを適用
    hz = (sympy.sympify('z**(-2)*((z+1)/2)**4')*h0z).expand()

    # scaling係数を取得
    scaling_coeff = [hz.coeff('z',k) for k in range(-3,3+1)]

    # 共役複素数解からなる因子を使用して多項式を作成
    d_h0z = sympy.sympify('sqrt(2.0)*(y-y1)/y1*(y-y2)/y2') \
                 .subs({'y':'-1/4*z+1/2-1/4/z', 'y1':y[1], 'y2':y[2]})

    # vanising momentsを適用
    d_hz = (sympy.sympify('z**(-2)*((z+1)/2)**4')*d_h0z).expand()

    # dual scaling係数を取得
    d_scaling_coeff = [sympy.re(d_hz.coeff('z',k)) for k in range(-4,5)]

    # wavelet係数を取得
    wavelet_coeff = [s*(-1)**k for k,s in zip(range(-4,4+1), d_scaling_coeff)]

    # dual wavelet係数を取得
    d_wavelet_coeff = [s*(-1)**k for k,s in zip(range(-3,3+1), scaling_coeff) ]

    return scaling_coeff, d_scaling_coeff, wavelet_coeff, d_wavelet_coeff

def main():
    scaling, d_scaling, wavelet_coeff, d_wavelet_coeff= cdf_9_7()

    lines = []
    lines.append('# CDF 9/7 scaling coefficients\n')
    for i, h in enumerate(scaling):
        lines.append(f'{mpmath.nstr(h, 40, min_fixed=0)}\n')
    with open('cdf_9_7_scaling_coefficients.txt', 'w') as f:
        f.writelines(lines)

    lines = []
    lines.append('# CDF 9/7 dual scaling coefficients\n')
    for i, h in enumerate(d_scaling):
        lines.append(f'{mpmath.nstr(h, 40, min_fixed=0)}\n')
    with open('cdf_9_7_dual_scaling_coefficients.txt', 'w') as f:
        f.writelines(lines)

    phi1_x, phi1_y, psi1_x, psi1_y = dual_cascade(scaling, wavelet_coeff)
    phi2_x, phi2_y, psi2_x, psi2_y = dual_cascade(d_scaling, d_wavelet_coeff)

    plt.plot(phi1_x, phi1_y)
    plt.plot(phi2_x, phi2_y)
    plt.grid()
    plt.legend(['CDF9/7 scaling', 'CDF9/7 dual scaling'])
    plt.savefig('cdf_9_7_scaling.png')
    plt.clf()

    plt.plot(psi1_x, psi1_y)
    plt.plot(psi2_x, psi2_y)
    plt.grid()
    plt.legend(['CDF9/7 wavelet', 'CDF9/7 dual wavelet'])
    plt.savefig('cdf_9_7_wavelet.png')
    plt.clf()

if __name__ == '__main__':
    main()
