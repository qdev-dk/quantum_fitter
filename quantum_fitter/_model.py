import lmfit
import matplotlib.pyplot as plt
import numpy as np


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = np.eig(np.dot(np.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


class ComplexResonatorModel(lmfit.model.Model):
    """
    The code below is from https://lmfit.github.io/lmfit-py/examples/
    example_complex_resonator_model.html#sphx-glr-download-examples-example-complex-resonator-model-py
    """
    __doc__ = "Complex Resonator model" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(ComplexResonatorModel.linear_resonator, *args, **kwargs)

        self.set_param_hint('Q', min=0)  # Enforce Q is positive

    def guess(self, data, f=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if f is None:
            return
        argmin_s21 = np.abs(data).argmin()
        fmin = f.min()
        fmax = f.max()
        f_0_guess = f[argmin_s21]  # guess that the resonance is the lowest point
        Q_min = 0.1 * (f_0_guess / (
                    fmax - fmin))  # assume the user isn't trying to fit just a small part of a resonance curve.
        delta_f = np.diff(f)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = f_0_guess / min_delta_f  # assume data actually samples the resonance reasonably
        Q_guess = np.sqrt(Q_min * Q_max)  # geometric mean, why not?
        Q_e_real_guess = Q_guess / (1 - np.abs(data[argmin_s21]))
        if verbose:
            print("fmin=", fmin, "fmax=", fmax, "f_0_guess=", f_0_guess)
            print("Qmin=", Q_min, "Q_max=", Q_max, "Q_guess=", Q_guess, "Q_e_real_guess=", Q_e_real_guess)
        params = self.make_params(Q=Q_guess, Q_e_real=Q_e_real_guess, Q_e_imag=0, f_0=f_0_guess)
        params['%sQ' % self.prefix].set(min=Q_min, max=Q_max)
        params['%sf_0' % self.prefix].set(min=fmin, max=fmax)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    @staticmethod
    def linear_resonator(x, f_0, Q, Q_e_real, Q_e_imag):
        Q_e = Q_e_real + 1j * Q_e_imag
        return 1 - (Q * Q_e ** -1 / (1 + 2j * Q * (x - f_0) / f_0))


class ResonatorModel(lmfit.model.Model):
    __doc__ = "Resonator model" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(ResonatorModel.S21funct, *args, **kwargs)

    def guess(self, data, f=None, linecomp=None, **kwargs):

        verbose = kwargs.pop('verbose', None)
        A_guess = np.mean(np.abs(data)[[0, -1]])
        S21_norm = data / A_guess
        f0_guess = f[np.argmin(np.abs(data))]
        if f is None:
            return
        a_guess = ResonatorModel.fitEllipse(np.real(1 / S21_norm), np.imag(1 / S21_norm))
        axes_guess = ResonatorModel.ellipse_axis_length(a_guess)
        D_guess = np.max(axes_guess[[0, 1]])
        Qi_guess = 2 * (f0_guess / ResonatorModel.fwhm(f, S21_norm))
        guess = [f0_guess, Qi_guess, Qi_guess/(2*D_guess), 0, A_guess, 0, 0, np.angle(np.mean(S21_norm))]
        Qe_mag_guess = guess[2]

        if verbose:
            print("fmin=", f.min(), "fmax=", f.max(), "f_0_guess=", f0_guess)
            print("Qmin=", 0, "Q_max=", 20*guess[2], "Q_guess=", Qi_guess)

        params = self.make_params(f_0=guess[0], Qi=guess[1], Qe_mag=guess[2], Qe_theta=guess[3],
                                  A=guess[4], alpha=guess[5], phi1=guess[6], phi2=guess[7])
        print(guess)

        params['%sf_0' % self.prefix].set(min=0.9 * f0_guess, max=1.1 * f0_guess)
        params['%sQi' % self.prefix].set(min=0, max=20*Qi_guess)
        params['%sQe_mag' % self.prefix].set(min=0, max=20*Qe_mag_guess)
        params['%sQe_theta' % self.prefix].set(min=-np.pi, max=np.pi)
        params['%sA' % self.prefix].set(min=0, max=2*A_guess)
        params['%salpha' % self.prefix].set(min=-np.inf, max=np.inf)
        params['%sphi1' % self.prefix].set(min=-np.inf, max=np.inf)
        params['%sphi2' % self.prefix].set(min=-np.inf, max=np.inf)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    @staticmethod
    def fitEllipse(x, y):
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2;
        C[1, 1] = -1
        E, V = np.linalg.eig(np.dot(np.linalg.pinv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    @staticmethod
    def ellipse_axis_length(a):
        b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        res1 = np.sqrt(abs(up / down1))
        res2 = np.sqrt(abs(up / down2))
        return np.array([res1, res2])

    # Find FWHM
    @staticmethod
    def fwhm(f, S21):
        mag = np.abs(S21)
        mag_half_max = (mag.min() + np.mean([mag[[0, -1]]])) / 2
        f0_idx = np.argmin(mag)

        half_max_left = np.abs((mag[0:f0_idx] - mag_half_max)).argmin()
        half_max_right = np.abs((mag[f0_idx::] - mag_half_max)).argmin()
        return abs(f[half_max_right + f0_idx] - f[half_max_left])

    # Full function for S21 response of the resonator.
    @staticmethod
    def S21funct(x, f_0, Qi, Qe_mag, Qe_theta, A, alpha, phi1, phi2):
        Qi *= 1e3
        Qe_mag *= 1e3
        Qe = Qe_mag * np.exp(-1j * Qe_theta)
        Qc = 1 / (np.real(1 / Qe))
        Q = 1 / (1 / Qc + 1 / Qi)
        S = A * (1 + alpha * (x - f_0) / f_0) * (1 - (Q / Qe) / (1 + 2 * 1j * Q * (x - f_0) / f_0)) * np.exp(
            1j * (phi1 * x + phi2))
        return S

