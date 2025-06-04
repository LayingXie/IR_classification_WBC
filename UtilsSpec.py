import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as interp

def unifxaxis(spec, xaxis, wl_pump, dointerp = None):
    """
    Parameters
    ----------
    spec : numpy array
        Input spectra.
    xaxis : numpy array
        original xaxis.
    wl_pump : int
        Pump Wavelength.
    dointerp : Logical, optional
        Variable indicates if interpolation is needed (only 2 x pixelNb). The default is None.

    Returns
    -------
    myVar : list
        Return list of variables including original spectra, original xaxis, Jacobian transposed spectra, etc.

    """
    myVar = {}
    if dointerp!=None:
        if np.all(spec<0)==True:
            myVar['Absolute_wavenumber'] = 1e7/wl_pump - xaxis
            myVar['Wavelength'] = 1e7/myVar['Absolute_wavenumber']
            myVar['SpectrumJacTRans'] = spec
            myVar['OrigSpec'] = np.power(myVar['Absolute_wavenumber'],2) * spec
            myVar['relative_wavenumber'] = myVar['Absolute_wavenumber'] - (1e7/wl_pump)
            xaxis = myVar['Wavelength']
            spec = myVar['OrigSpec']
        else:
            xaxis = xaxis
            spec = spec    
        # Interpolate on wavelength axis
        fn_interp = interp.interp1d(xaxis,spec)
        newaxis_wl = []
        myVar['OrigSpec'] = spec
        myVar['Wavelength'] = xaxis
        for i in range(xaxis.size-1):
            newaxis_wl = np.append(newaxis_wl, [xaxis[i], (xaxis[i]+xaxis[i+1])/2])
        newaxis_wl = np.append(newaxis_wl, xaxis[xaxis.size-1])
        spec_wl = fn_interp(newaxis_wl)
        xaxis_wl = np.append(newaxis_wl, newaxis_wl[newaxis_wl.size-1] + 
                             round(newaxis_wl[newaxis_wl.size-1]-newaxis_wl[newaxis_wl.size-2],2))
        spec_wl= np.append(spec_wl,spec_wl[spec_wl.size-1])
        myVar['InterpolateSpec_wl'] = spec_wl
        myVar['InterpolateWavelength'] = xaxis_wl
        myVar['SpectrumJacTRans_wlinterp'] = spec_wl / np.power(xaxis_wl,2)
        myVar['Absolute_wavenumber_wlinterp'] = 1e7/xaxis_wl
        myVar['Relative_wavenumber_wlinterp'] = 1e7/xaxis_wl - (1e7/wl_pump)

        # Interpolate on wavenumber axis
        myVar['SpectrumJacTRans_wninterp'] = spec / np.power(xaxis,2)
        myVar['Absolute_wavenumber'] = 1e7/xaxis
        fn_interpwn = interp.interp1d(myVar['Absolute_wavenumber'], myVar['SpectrumJacTRans_wninterp'])
        newaxis_abswn = []
        for i in range(xaxis.size-1):
            newaxis_abswn = np.append(newaxis_abswn, [myVar['Absolute_wavenumber'][i], 
                                                      (myVar['Absolute_wavenumber'][i] + myVar['Absolute_wavenumber'][i+1])/2])
        newaxis_abswn = np.append(newaxis_abswn, myVar['Absolute_wavenumber'][xaxis.size-1])
        spec_wn = fn_interpwn(newaxis_abswn)
        xaxis_wn = np.append(newaxis_abswn, newaxis_abswn[newaxis_abswn.size-1] + 
                             round(newaxis_abswn[newaxis_abswn.size-1]-newaxis_abswn[newaxis_abswn.size-2],2))
        spec_wn = np.append(spec_wn, spec_wn[spec_wn.size-1])
        myVar['InterpolateSpec_wn'] = spec_wn
        myVar['Interpolate_abswn'] = xaxis_wn
        myVar['Relative_wavenumber_wninterp'] = xaxis_wn - (1e7/wl_pump)
    else:
        if np.all(xaxis<0)==True:
            myVar['Absolute_wavenumber'] = 1e7/wl_pump - xaxis
            myVar['Wavelength'] = 1e7/myVar['Absolute_wavenumber']
            myVar['SpectrumJacTRans'] = spec
            myVar['OrigSpec'] = np.power(myVar['Absolute_wavenumber'],2) * spec
            myVar['relative_wavenumber'] = myVar['Absolute_wavenumber'] - (1e7/wl_pump)
        else:        
            myVar['OrigSpec'] = spec
            myVar['Wavelength'] = xaxis
            myVar['SpectrumJacTRans'] = spec / np.power(xaxis,2)
            myVar['Absolute_wavenumber'] = 1e7/xaxis
            myVar['relative_wavenumber'] = myVar['Absolute_wavenumber'] - (1e7/wl_pump)  
    
    return myVar

def modified_z_scores(spc):
    """
    Parameters
    ----------
    spc : numpy ndarray
        Input data.

    Returns
    -------
    modified_z_scores : numpy ndarray
        Modified data.

    """
    median_int = np.median(spc)
    mad_int = np.median([np.abs(spc - median_int)])
    modified_z_scores = 0.6745 * (spc - median_int) / mad_int
    return modified_z_scores

def interpfunc(xrange, data, oldX, newX):
    """
    Parameters
    ----------
    xrange : numpy array of two values
        Mininimum and maximum value of old x-axis.
    data : numpy array
        Input data.
    oldX : int
        length of the old data.
    newX : int
        Length for the interpolation.

    Returns
    -------
    newdata : numpy array
        Interpolated data.

    """
    x_old = np.linspace(xrange[0], xrange[1], oldX)
    x_new = np.linspace(xrange[0], xrange[1], newX)
    if data.shape[0] != 1:
        newdata = np.zeros((data.shape[0], newX))
        for i in range(data.shape[0]):
            TMPout = interp.interp1d(x_old, data[i,:], axis=0)
            newdata[i,:] = TMPout(x_new)
    else:
        TMPout = interp.interp1d(x_old, data, axis=0)
        newdata = TMPout(x_new)    
    return newdata

def peak_fit(x, y, mean, sd, Type="Gauss"):
    """
    Parameters
    ----------
    x : numpy array
        Input xaxis.
    y : numpy array
        Input yvalue.
    mean : numpy array
        mean.
    sd : numpy array
        standard deviation.
    Type : Logical, optional
        Fitting type including "Gauss" and "Lorentz". The default is "Gauss".

    Returns
    -------
    result : list
        List of the fitting parameters.

    """
    O_0 = np.min(y)
    b_0 = mean
    c_0 = sd
    A_0 = np.max(y) - np.min(y)
    if Type == "Gauss":
        p0 = [O_0, A_0, b_0, c_0]
        popt, _ = curve_fit(gauss_func, x, y, p0=p0, method='trf', bounds=(0, np.inf))
        result = {'O': popt[0], 'A': popt[1], 'b': popt[2], 'c': popt[3]}       
    elif Type == "Lorentz":
        p0 = [O_0, A_0, b_0, c_0]
        popt, _ = curve_fit(lorentz_func, x, y, p0=p0, method='trf', bounds=(0, np.inf))
        result = {'O': popt[0], 'A': popt[1], 'b': popt[2], 'c': popt[3]} 
    else:
        # handle the case where Type is neither "Gauss" nor "Lorentz"
        print(f"Invalid Type: {Type}")
        return None
    return result

def fit_degree(degree, Peaks_tabled, Peaks_found, x_old):
    """
    Parameters
    ----------
    degree : int
        polynomial degree.
    Peaks_tabled : numpy array
        Array with peaks values.
    Peaks_found : numpy array
        Array with founded peaks.
    x_old : numpy array
        xaxis input.

    Returns
    -------
    x : numpy array
        Fitted results.

    """
    if degree == 0:
        p0 = [1]
    else:
        p0 = [1] * (degree + 1)  
    popt, _ = curve_fit(polynomial_func, Peaks_found, Peaks_tabled, p0=p0, method='trf', bounds=(0, np.inf))
    COEF = popt.tolist() 
    x = polynomial_func(x_old, *COEF)
    return x

def gauss_func(x, O, A, b, c):
    """
    Parameters
    ----------
    x : numpy array
        input xaxis.
    O, A, b, c : int
        Initialization values.

    Returns
    -------
    numpy array
        Corresponding values using Gaussian function.
    """
    return O + A * np.exp(-(x - b) ** 2 / (2 * c ** 2))
   
def lorentz_func(x, O, A, b, c):
    """
    Parameters
    ----------
    x : numpy array
        input xaxis.
    O, A, b, c : int
        Initialization values.

    Returns
    -------
    numpy array
        Corresponding values using Lorentzian function.

    """
    return O + A * c / ((c ** 2) + (b - x) ** 2)

def polynomial_func(x, *coefficients):
    """
    Parameters
    ----------
    x : numpy array
        input xaxis.
    *coefficients : list
        list of coefficients.

    Returns
    -------
    numpy array
        Corresponding values using polynomial function.

    """
    return np.sum([coefficients[i] * x ** i for i in range(len(coefficients))], axis=0)
