from __future__ import division
from pylab import *
from scipy import optimize
from scipy import stats

def curve_fit(function, xdata, ydata, initial, u_ydata=None, return_covariance=False, function_kwargs={}, **curve_fit_kwargs):
    """Wrapper for optimize.curve_fit that computes the uncertainty in fit parameters"""

    if isinstance(initial, dict):
        initial_names = initial.keys()
        initial_vals = initial.values()
        def function_wrapper(x, *params):
            kwargs = {name: val for name, val in zip(initial_names, params)}
            kwargs.update(function_kwargs)
            return function(x, **kwargs)
    else:
        initial_vals = initial
        def function_wrapper(x, *params):
            return function(x, *params, **function_kwargs)

    params, covariance = optimize.curve_fit(function_wrapper, xdata, ydata, initial_vals, u_ydata, **curve_fit_kwargs)
    if not isinstance(covariance,ndarray):
        raise RuntimeError('uncertainties in fit parameters are infinity: fit did not converge.')
    u_params = [sqrt(abs(covariance[i,i])) for i in range(len(params))]
    if u_ydata is not None:
        chi2 = ((ydata - function(xdata,*params))**2/u_ydata**2).sum()
        df = (len(xdata)-len(params))
        reduced_chi2 = chi2/df
        p_value_of_bad_fit = 1 - stats.chi2.cdf(chi2, df)
        p_value_of_overly_good_fit = 1 - p_value_of_bad_fit
        print 'chi squared per degree of freedom:',  reduced_chi2
        print 'p_value_of_bad_fit', p_value_of_bad_fit
        print 'p_value_of_overly_good_fit', p_value_of_overly_good_fit
        if p_value_of_bad_fit < 0.05 or p_value_of_overly_good_fit < 0.05:
            print 'bad fit! Scaling uncertainties by a factor of', sqrt(reduced_chi2)
        curve_fit(function, xdata, ydata, initial, u_ydata*sqrt(reduced_chi2), return_covariance, **kwargs)

    if isinstance(initial, dict):
        params = dict(zip(initial_names, params))
        u_params = dict(zip(['u_'+name for name in initial_names], u_params))

    if return_covariance:
        return params, u_params, covariance
    else:
        return params, u_params

def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param/1e6
        params_with_partial_differential = zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(x, *params_with_partial_differential)
        partial_derivative = (model_at_partial_differential - model_at_center)/d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives

def model_uncertainty(function, x, params, covariance):
    u_params = [sqrt(abs(covariance[i,i]))  for i in range(len(params))]
    model_partial_derivatives = partial_derivatives(function, x, params, u_params)
    try:
        squared_model_uncertainty = zeros(x.shape)
    except TypeError:
        squared_model_uncertainty = 0
    for i in range(len(params)):
        for j in range(len(params)):
            squared_model_uncertainty += model_partial_derivatives[i]*model_partial_derivatives[j]*covariance[i,j]
    return sqrt(squared_model_uncertainty)

def model_shaded_uncertainty(function, x, params, covariance, yrange=None, resolution=1024, columns_normalised=False):
    model_mean = function(x, *params)
    model_stddev = model_uncertainty(function, x, params, covariance)
    if yrange is None:
        yrange = [(model_mean - 10*model_stddev).min(), (model_mean + 10*model_stddev).max()]
    y = linspace(yrange[0], yrange[1], resolution)
    Model_Mean, Y = meshgrid(model_mean, y)
    Model_Stddev, Y = meshgrid(model_stddev, y)
    if columns_normalised:
        probability = exp(-(Y - Model_Mean)**2/(2*Model_Stddev**2))
    else:
        probability = normpdf(Y, Model_Mean, Model_Stddev)
#    probability = (abs(Y - Model_Mean)/Model_Stddev).clip(0,3)
    return probability, [x.min(), x.max(), y.min(), y.max()]

# Linear function
def linear(x, m, c=0):
    return m*x + c

def fit_linear(xdata, ydata, u_ydata=None, force_through_zero=False, **kwargs):
    m_guess = (ydata[-1] - ydata[0])/(xdata[-1] - xdata[0])
    c_guess = ydata[0] - m_guess*xdata[0]
    if force_through_zero:
        initial_guesses = [m_guess]  # ignore c, defaults to zero
    else:
        initial_guesses = [m_guess, c_guess]
    return curve_fit(linear, xdata, ydata, initial_guesses, u_ydata, **kwargs)

# Quadratic function
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def fit_quadratic(xdata, ydata, u_ydata=None, **kwargs):
    return curve_fit(quadratic, xdata, ydata, [1.0,1.0,1.0],u_ydata, **kwargs)

# Exponential step function
def stepfn(t, initial, final, step_time, rise_time):
    # 10-90% rise time, approx sqrt(2)*log(9)/5 ~= 0.62 x (1/e^2 integration width):
    exponent = -4*log(3)*(t-step_time)/rise_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent,700])
    return (final - initial)/(1 + exp(exponent)) + initial

def fit_step(xdata, ydata, u_ydata=None, pars_guess=None, **kwargs):
    if not pars_guess:
        return curve_fit(stepfn, xdata, ydata,
                         [ydata[0], ydata[-1], (xdata[-1]+xdata[0])/2, (xdata[-1]+xdata[0])/10],
                         u_ydata, **kwargs)
    else:
        return curve_fit(stepfn, xdata, ydata, pars_guess, u_ydata, **kwargs)

# Lorentzian function
def lorentzian(x, amplitude, x0, fwhm, offset):
    return amplitude/(1 + (x-x0)**2/(fwhm/2)**2) + offset

def fit_lorentzian(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        ymin = min(ydata)
        ymax = max(ydata)
        amplitude_guess = ymax - ymin
        # check for an inverted peak
        if abs(mean(ydata[[0,-1]]) - ymin) > abs(mean(ydata[[0,-1]]) - ymax) or True:
            prob = ydata-ymin
            offset_guess = ymin
        else:
            prob = ymax-ydata
            offset_guess = ymax
            amplitude_guess = -amplitude_guess
        x0_guess = sum(xdata*prob)/sum(prob)
        fwhm_guess = sum(abs(xdata-x0_guess)*prob)/sum(prob)
        pars_guess = [amplitude_guess,x0_guess,fwhm_guess,offset_guess]
    return curve_fit(lorentzian, xdata, ydata, pars_guess, u_ydata, **kwargs)

# Gaussian function
def gaussian(x, amplitude, x0, fwhm, offset):
    return amplitude*exp(-4*log(2)*(x-x0)**2/fwhm**2) + offset

def fit_gaussian(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        ymin = min(ydata)
        ymax = max(ydata)
        amplitude_guess = ymax - ymin
        # check for an inverted peak
        if abs(mean(ydata[[0,-1]]) - ymin) > abs(mean(ydata[[0,-1]]) - ymax):
            prob = ydata-ymin
            offset_guess = ymin
        else:
            prob = ymax-ydata
            offset_guess = ymax
            amplitude_guess = -amplitude_guess
        x0_guess = sum(xdata*prob)/sum(prob)
        fwhm_guess = sum(abs(xdata-x0_guess)*prob)/sum(prob)
        pars_guess = [amplitude_guess,x0_guess,fwhm_guess,offset_guess]
    return curve_fit(gaussian, xdata, ydata, pars_guess, u_ydata, **kwargs)

def laguerre_gauss_01(x, amplitude, x0, fwhm, offset):
    return amplitude*((x-x0)**2)*exp(-4*log(2)*(x-x0)**2/fwhm**2) + offset

def fit_laguerre_gauss_01(xdata, ydata, u_ydata=None, **kwargs):
    ymin = min(ydata)
    ymax = max(ydata)
    amplitude_guess = ymax - ymin
    # check for an inverted peak
    if abs(mean(ydata[[0,-1]]) - ymin) > abs(mean(ydata[[0,-1]]) - ymax):
        prob = ydata-ymin
        offset_guess = ymin
    else:
        prob = ymax-ydata
        offset_guess = 0#ymax
        amplitude_guess = amplitude_guess
    x0_guess = sum(xdata*prob)/sum(prob)
    fwhm_guess = sum(abs(xdata-x0_guess)*prob)/sum(prob)
    return curve_fit(laguerre_gauss_01, xdata, ydata, [amplitude_guess,x0_guess,fwhm_guess,offset_guess], u_ydata, **kwargs)

# Thomas-Fermi profile
def thomas_fermi(x, amplitude, x0, xTF, offset):
    if iterable(x):
        return array([amplitude*max(1-(xi-x0)**2/xTF**2, 0)**(3/2) + offset for xi in x])
    else:
        return amplitude*max(1-(xi-x0)**2/xTF**2, 0)**(3/2) + offset

def fit_thomas_fermi(xdata, ydata, u_ydata=None, **kwargs):
    ymin = min(ydata)
    ymax = max(ydata)
    amplitude_guess = ymax - ymin
    if abs(mean(ydata[[0,-1]]) - ymin) < abs(mean(ydata[[0,-1]]) - ymax):
        prob = ydata-ymin
        offset_guess = ymin
    else:
        prob = ymax-ydata
        offset_guess = ymax
        amplitude_guess = -amplitude_guess
    x0_guess = sum(xdata*prob)/sum(prob)
    xTF_guess = sum(abs(xdata-x0_guess)*prob)/sum(prob)
    return curve_fit(thomas_fermi, xdata, ydata, [amplitude_guess,x0_guess,xTF_guess,offset_guess], u_ydata, **kwargs)

# Exponential rise or decay from initial to final value (e.g. MOT load)
def expfn(t, initial, final, decay_time, zero_time=False):
    if zero_time:
        exponent = -(t-t.min())/decay_time
    else:
        exponent = -t/decay_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent,700])
    return (final - initial)*(1 - exp(exponent)) + initial

def log_expfn(t, initial, final, decay_time):
    exponent = -t/decay_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent,700])
    return log((final - initial)*(1 - exp(exponent)) + initial)

def fit_exp(xdata, ydata, u_ydata=None, fix_initial=False, fix_final=False, initial_val=0, final_val=0, **kwargs):
    initial_guess = max(ydata)
    final_guess =  min(ydata)
    decay_time_guess =  (max(xdata)-min(xdata))/2
    if not fix_initial and not fix_final:
        return curve_fit(expfn, xdata, ydata, [initial_guess, final_guess, decay_time_guess], u_ydata, **kwargs)
    elif not fix_initial and fix_final:
        def fitfn(t, initial, decay_time):
            return expfn(t, initial, final_val, decay_time)
        return curve_fit(fitfn, xdata, ydata, [initial_guess, decay_time_guess], u_ydata, **kwargs)
    elif fix_initial and not fix_final:
        def fitfn(t, final, decay_time):
            return expfn(t, initial_val, final, decay_time)
        return curve_fit(fitfn, xdata, ydata, [final_guess, decay_time_guess], u_ydata, **kwargs)
    else:
        def fitfn(t, decay_time):
            return expfn(t, initial_val, final_val, decay_time)
        return curve_fit(fitfn, xdata, ydata, [decay_time_guess], u_ydata, **kwargs)

def exp_initial_rate(initial, final, decay_time, d_initial, d_final, d_decay_time):
    rate = (final-initial)/decay_time
    d_rate = rate * sqrt( (d_initial**2 + d_final**2)/(final-initial)**2 + d_decay_time**2 / decay_time**2)
    return rate, d_rate

def fit_log_exp(xdata, ydata, u_ydata=None, **kwargs):
    initial_guess = ydata[0]
    final_guess =  ydata[-1]
    decay_time_guess =  (xdata[-1]-xdata[0])/2
    return curve_fit(log_expfn, xdata, log(ydata), [initial_guess, final_guess, decay_time_guess], u_ydata, **kwargs)

# Rabi oscillations (in terms of total spin projection Fz)
def rabi(t, f, f0, fR, A=1, c=0):
    det = f-f0
    Fz = -(det**2 + fR**2*cos(2*pi*t*sqrt(det**2+fR**2)))/(det**2 + fR**2)
    return A*Fz + c

def fit_rabi(xdata, ydata, t_pulse, scaling=True, offset=True, **kwargs):
    prob = ydata - min(ydata)
    f0_guess = sum(xdata*prob)/sum(prob)
    fR_guess = 1/(2*t_pulse)
    params_guess = [f0_guess, fR_guess]
    if scaling:
        A_guess = 1
        params_guess.append(A_guess)
    if offset:
        c_guess = 0
        params_guess.append(c_guess)
    def fitfn(f, *pars):
        return rabi(t_pulse, f, *pars)
    x = curve_fit(fitfn, xdata, ydata, params_guess, **kwargs)
    return x

def fit_rabi_t(xdata, ydata, f_pulse, scaling=True, offset=True, **kwargs):
    from scipy.fftpack import rfft, rfftfreq
    # ordering = xdata.argsort()
    ordering = argsort(xdata).values
    yhat = rfft(ydata[ordering])
    idx = (yhat**2).argmax()
    freqs = rfftfreq(len(xdata), d=xdata[ordering].diff()[1])
    #fR_guess = 0.7*freqs[idx]
    fR_guess = 2/max(xdata) # assuming roughly one Rabi period of data
    # fR_guess = 1/(19e-6)
    f0_guess = f_pulse
    params_guess = [f0_guess, fR_guess]
    # print params_guess
    A_guess = 1
    if scaling*0:
        params_guess.append(A_guess)
    if scaling and offset:
        c_guess = 0
        params_guess.append(c_guess)
    def fitfn(t, *pars):
        return rabi(t, f_pulse, *pars)
    params, u_params = curve_fit(fitfn, xdata, ydata, params_guess, **kwargs)
    params[1] = abs(params[1])
    return params, u_params

def fit_rabi_pops(xdata, ydata, f_pulse, state=0, offset=True, **kwargs):
    fR_guess = 1/max(xdata) # assuming roughly one Rabi period of data
    f0_guess = f_pulse
    A_guess = max(ydata)
    params_guess = [f0_guess, fR_guess, A_guess]
    c_guess = min(ydata)
    if offset:
        params_guess.append(c_guess)
    def populations(t, f_pulse, f0, fR):
        P0 = 0.5*(rabi(t, f_pulse, f0, fR)+1)
        P1 = 1-P0
        return P0, P1
    def rabiN(t, f_pulse, f0, fR, A=A_guess, c=c_guess):
        return A*populations(t, f_pulse, f0, fR)[state]+c
    def fitfn(t, *pars):
        return rabiN(t, f_pulse, *pars)
    return curve_fit(fitfn, xdata, ydata, params_guess, **kwargs)

# Sinusoidal function
def sine(t, f, A=1, c=0, phi=0):
    return A*sin(2*pi*f*t+phi)+c

def fit_sine(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        f_guess = 1/max(xdata) # assuming roughly one period of data
        A_guess = (max(ydata) - min(ydata))/2
        c_guess = mean(ydata)
        phi_guess = 0
        pars_guess = [f_guess, A_guess, c_guess, phi_guess]
    return curve_fit(sine, xdata, ydata, pars_guess, u_ydata, **kwargs)

# Phase domain Ramsey
def cos_ramsey(phi, A=1, c=0, dphi=0):
    # Note: phi is in degrees
    return A*cos(pi*phi/180+dphi)+c

def fit_cos_ramsey(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        A_guess = (max(ydata) - min(ydata))/2
        c_guess = mean(ydata)
        dphi_guess = 0
        pars_guess = [A_guess, c_guess, dphi_guess]
    params, u_params = curve_fit(cos_ramsey, xdata, ydata, pars_guess, u_ydata, **kwargs)
    if params[0] < 0:
        params[0] *= -1
        params[2] -= pi
    return params, u_params

# AC line noise
def line_noise(t, amps, phases, offset=0, gradient=0):
    freqs = 50*(arange(len(amps))+1)
    harmonics = array([A*sin(2*pi*f*t+phi) for (A, f, phi) in zip(amps, freqs, phases)])
    return harmonics.sum(axis=0)+offset+gradient*t

def fit_line_noise(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    def fitfn(t, *pars):
        n = 2
        m = len(pars)-n
        amps = pars[n:n+int(m/2)]
        phases = pars[n+int(m/2):]
        offset = pars[0]
        gradient = pars[1]
        return line_noise(t, amps, phases, offset, gradient)
    return curve_fit(fitfn, xdata, ydata, pars_guess, u_ydata, **kwargs)

# Cosine with exponential decay
def sine_decay(t, f, A=1, c=0, phi=0, tc=1e9):
    return A*exp(-t/tc)*sin(2*pi*f*t+phi)+c

def fit_sine_decay(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        f_guess = 1/max(xdata) # assuming roughly one period of data
        A_guess = (max(ydata) - min(ydata))/2
        c_guess = mean(ydata)
        phi_guess = pi/2
        tc_guess = max(xdata) # assuming roughly one 1/e time of data
        pars_guess = array([f_guess, A_guess, c_guess, phi_guess, tc_guess])
        pars_guess = pars_guess.round(10)
    return curve_fit(sine_decay, xdata, ydata, pars_guess, u_ydata, **kwargs)

# Ramsey function
def ramsey(t, q, A, f, tc, c=0, phi=0):
    # return -A*exp(-t/tc)*cos(2*pi*f*t+phi)+c
    return -A*exp(-t**2/tc**2)*cos(2*pi*q*t)*cos(2*pi*f*t+phi)+c

def fit_ramsey(xdata, ydata, q=0, A=0, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        f_guess = 2/max(xdata) # assuming roughly two periods of data
        tc_guess = 10*max(xdata)
        A_guess = (max(ydata) - min(ydata))/2
        c_guess = mean(ydata)
        phi_guess = 0
        pars_guess = array([A_guess, f_guess, tc_guess, c_guess, phi_guess])
        pars_guess = pars_guess.round(10)
        if A:
            pars_guess = pars_guess[1:]
    if A:
        def fitfn(t, *pars):
            return ramsey(t, q, A, *pars)
    else:
        def fitfn(t, *pars):
            return ramsey(t, q, *pars)
    return curve_fit(fitfn, xdata, ydata, pars_guess, u_ydata, **kwargs)

# sinc squared function
def sincfn(x, amplitude, x0, fwhm, offset):
    return amplitude * sinc((x-x0)/(fwhm/2))**2 + offset

def fit_sinc(xdata, ydata, pars_guess=None, u_ydata=None, **kwargs):
    if not pars_guess:
        ymin = min(ydata)
        ymax = max(ydata)
        amplitude_guess = ymax - ymin
        # check for an inverted peak
        if abs(mean(ydata[[0, -1]]) - ymin) > abs(mean(ydata[[0, -1]]) - ymax) or True:
            prob = ydata-ymin
            offset_guess = ymin
        else:
            prob = ymax-ydata
            offset_guess = ymax
            amplitude_guess = -amplitude_guess
        x0_guess = sum(xdata*prob)/sum(prob)
        fwhm_guess = sum(abs(xdata-x0_guess)*prob)/sum(prob)
        pars_guess = [amplitude_guess,x0_guess,fwhm_guess,offset_guess]
    return curve_fit(sincfn, xdata, ydata, pars_guess, u_ydata, **kwargs)


# EXAMPLES
def step_example():
    t = linspace(0,10,100000)
    y = zeros(len(t))
    y[t > 5] = 5
    fy = fft(y)
    fy[10:] = 0
    y = ifft(fy).real
    y += normal(size=len(t))

    (initial, final, step_time, rise_time), (d_initial, d_final, d_step_time, d_rise_time) = fit_step(t,y)
    print 'initial = %f +/- %f' % (initial, d_initial)
    print 'final = %f +/- %f' % (final, d_final)
    print 'step_time= %f +/- %f' % (step_time, d_step_time)
    print 'rise_time = %f +/- %f' % (rise_time, d_rise_time)

    plot(t,y)
    plot(t,stepfn(t, initial, final, step_time, rise_time), linewidth = 5)
    grid(True)
    show()

def linear_example():
    t = arange(10)
    y = 2*t + 5 + rand(len(t))
    params, u_params = fit_linear(t,y)
    print params, u_params
    plot(t,y,'o')
    plot(t,linear(t, **params))
    show()


def expexample():
    x = linspace(0,10,50)
    y = 3*exp(-2*x) + 1 + rand(len(x))
    params, u_params = fit_exp(x,y)
    print params
    plot(x,y,'o')
    plot(x, expfn(x,*params))
    show()

# Renamed these functions, keeping old names for backwards compatibility:
expRise = expfn
fit_expRise = fit_exp
expRise_initial_rate = exp_initial_rate

#if __name__ == '__main__':
    #step_example()
    #linear_example()
    #expexample()
