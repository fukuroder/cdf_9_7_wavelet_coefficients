import numpy as np
import scipy.signal
import sympy
import mpmath
import matplotlib.pyplot as plt

# precition
mpmath.mp.prec = 256

def cascade_wavelet(phi, h, J):
    div = 2**J

    # get wavelet coefficients
    start = -(len(h)-1)//2
    g = [s*(-1)**k for k,s in enumerate(h, start)]

    # make wavelet
    x = np.linspace(-3, 4, num=div*7, endpoint=False)
    psi = np.zeros(div*7)
    for k, gk in enumerate(g):
        for i in range(len(psi)):
            if 0 <= 2*i-div*k < len(phi):
                psi[i] += np.sqrt(2)*gk*phi[2*i-div*k]
    return x, psi

def dual_cascade(h, dh, J=7):
    phi_x, phi_y, _ = scipy.signal.cascade(h, J=J)
    phi_x -= (len(h)-1)//2
    psi_x, psi_y = cascade_wavelet(phi_y, dh, J=J)
    return phi_x, phi_y, psi_x, psi_y

def cdf_9_7():
    # make polynomial
    # q(y) = 20y^3 + 10*y^2 + 4*y + 1
    qy = [mpmath.binomial(4-1+k,k) for k in reversed(range(0,4))]

    # get polynomial roots q(y)
    # y[0]: -0.3423840948
    # y[1]: -0.0788079525 + 0.3739306454j
    # y[2]: -0.0788079525 - 0.3739306454j
    y = mpmath.polyroots(qy)

    # make polynomial using real root
    y_real = [yk for yk in y if mpmath.im(yk) == 0]
    assert len(y_real) == 1
    h0z = sympy.sympify('-sqrt(2.0)*(y-y0)/y0') \
               .subs({'y':'-1/4*z+1/2-1/4/z', 'y0':y_real[0]})

    # adapt vanishing moments
    hz = (sympy.sympify('z**(-2)*((z+1)/2)**4')*h0z).expand()

    # get scaling coefficients
    scaling_coeff = [hz.coeff('z',k) for k in range(-3,3+1)]

    # make polynomial using complex root
    y_complex = [yk for yk in y if mpmath.im(yk) != 0]
    assert len(y_complex) == 2
    d_h0z = sympy.sympify('sqrt(2.0)*(y-y1)/y1*(y-y2)/y2') \
                 .subs({'y':'-1/4*z+1/2-1/4/z', 'y1':y_complex[0], 'y2':y_complex[1]})

    # adapt vanishing moments
    d_hz = (sympy.sympify('z**(-2)*((z+1)/2)**4')*d_h0z).expand()

    # get dual scaling coefficients
    d_scaling_coeff = [sympy.re(d_hz.coeff('z',k)) for k in range(-4,4+1)]

    return scaling_coeff, d_scaling_coeff

def main():
    # get CDF 9/7 scaling coeffients
    scaling, d_scaling= cdf_9_7()

    # write scaling coeffients
    lines = []
    lines.append('# CDF 9/7 scaling coefficients\n')
    for h in scaling:
        lines.append(f'{mpmath.nstr(h, 40, min_fixed=0)}\n')
    with open('cdf_9_7_scaling_coefficients.txt', 'w') as f:
        f.writelines(lines)

    # write dual scaling coeffients
    lines = []
    lines.append('# CDF 9/7 dual scaling coefficients\n')
    for h in d_scaling:
        lines.append(f'{mpmath.nstr(h, 40, min_fixed=0)}\n')
    with open('cdf_9_7_dual_scaling_coefficients.txt', 'w') as f:
        f.writelines(lines)

    # get an approximation of scaling function
    phi1_x, phi1_y, psi1_x, psi1_y = dual_cascade(scaling, d_scaling)
    phi2_x, phi2_y, psi2_x, psi2_y = dual_cascade(d_scaling, scaling)

    # plot scaling function
    plt.plot(phi1_x, phi1_y)
    plt.plot(phi2_x, phi2_y)
    plt.grid()
    plt.legend(['CDF9/7 scaling', 'CDF9/7 dual scaling'])
    plt.savefig('cdf_9_7_scaling.png')
    plt.clf()

    # plot wavelet
    plt.plot(psi1_x, psi1_y)
    plt.plot(psi2_x, psi2_y)
    plt.grid()
    plt.legend(['CDF9/7 wavelet', 'CDF9/7 dual wavelet'])
    plt.savefig('cdf_9_7_wavelet.png')
    plt.clf()

if __name__ == '__main__':
    main()
