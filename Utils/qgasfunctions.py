# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:47:08 2013

qgasfunctions: mathematical functions used in quantum gas experiments.  For example for fitting to data

@author: ispielma
"""

import numpy
import scipy
import scipy.optimize


# Functions for 1+D fiting using IanStyle fit functions: 
# 'xyVals' is a list of the independent variable arrays, 
# 'p' the parameter vector 
#

def gaussian1D(xyVals, p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]+p[1]*numpy.exp(-1*(xyVals[0]-p[2])**2/(2*p[3]**2));
    
def TF1D(xyVals, p) :
    # A Thomas Fermi peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   radius                    : p[3]
    condition = (1 - ((xyVals[0]-p[2])/p[3])**2);
    condition[condition < 0] = 0;
    return p[0] + (p[1]*condition)**(3/2);
    
def lorentzian(xyVals, p) :
    # A lorentzian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Full Width at Half Maximum   : p[3]
    return p[0]+(p[1]/numpy.pi)/(1.0+((xyVals[0]-p[2])/p[3])**2);
    
def line(xyVals,p) :
    # A linear fit with:
    #   Intercept                    : p[0]
    #   Slope                        : p[1]
    return p[0]+p[1]*xyVals[0];
    
def power(xyVals,p) :
    # A power law fit with:
    #   Normalization                : p[0]
    #   Offset                       : p[1]
    #   Constant                     : p[3]
    return p[0]*(xyVals[0]-p[1])**p[2]+p[3];


def gaussian2D(xyVals, p) :
    ''' A 2D gaussian peak with:
       Constant Background          : p[0]
       Peak height above background   : p[1]
       X Central value                : p[2]
       X Standard deviation           : p[3]
       Y Central value                : p[4]
       Y Standard deviation           : p[5] '''
    return p[0] + p[1]*numpy.exp(-1*(xyVals[0]-p[2])**2/(2*p[3]**2)-1*(xyVals[1]-p[4])**2/(2*p[5]**2));
    
def TF2D(xyVals, p) :
    ''' A 2D gaussian peak with:
       Constant Background            : p[0]
       Peak height above background   : p[1]
       X Central value                : p[2]
       X width                        : p[3]
       Y Central value                : p[4]
       Y width                        : p[5] '''
    condition = (1.0 - ((xyVals[0]-p[2])/p[3])**2.0 - ((xyVals[1]-p[4])/p[5])**2.0);
    condition[condition < 0.0] = 0.0;
    return p[0] + (p[1]*condition)**(3.0/2.0);

def TF_Thermal(xyVals, p):
	'''A 2D Tomas Fermi fit plus a thermal cloud, assume centers of both clouds overlap:
       Constant Background               : p[0]
       TF peak height above background   : p[1]
		Gaussian peak height              : p[2]
       X Central value                   : p[3]
       Y Central value                   : p[4]
       TF X width                        : p[5]
       TF Y width                        : p[6]
  	   Gauss X width                     : p[7]
       Gauss Y width                     : p[8] '''
	condition = (1.0 - ((xyVals[0]-p[3])/p[5])**2.0 - ((xyVals[1]-p[4])/p[6])**2.0);
	condition[condition < 0.0] = 0.0;
	return p[0] + (p[1]*condition)**(3.0/2.0)+p[0] + (p[2]*condition)**(3.0/2.0)+p[2]*numpy.exp(-1*(xyVals[0]-p[3])**2/(2*p[7]**2)-1*(xyVals[1]-p[4])**2/(2*p[8]**2));

def puck(xyVals, p) :
    # returns 1.0 within a specified boundary. 0.0 everywhere else:
    #   X Central value                : p[0]
    #   X width / 2                       : p[1]
    #   Y Central value                : p[2]
    #   Y width / 2                        : p[3]
    condition = (1.0 - ((xyVals[0]-p[0])/p[1])**2.0 - ((xyVals[1]-p[2])/p[3])**2.0);
    condition[condition < 0.0] = 0.0;
    condition[condition > 0.0] = 1.0;
    return condition;
    
def Thermal2DSG(xyVals, p) :
    # Three 2D gaussian peaks with specified displacements. widths are assumed the same:
    #   Peak height above background of mf = 0   : p[0]
    #   Peak height above background of mf = -1  : p[1]
    #   Peak height above background of mf = 1  : p[2]
    #   X width                        : p[3]
    #   Y width                         : p[4]

    xm = xyVals[2];
    xz = xyVals[3];
    xp = xyVals[4];

    total = p[0]*numpy.exp(-1*(xyVals[0]-xm[0])**2/(2*p[3]**2)-1*(xyVals[1]-xm[1])**2/(2*p[4]**2));
    total = total + p[1]*numpy.exp(-1*(xyVals[0]-xz[0])**2/(2*p[3]**2)-1*(xyVals[1]-xz[1])**2/(2*p[4]**2));
    total = total + p[2]*numpy.exp(-1*(xyVals[0]-xp[0])**2/(2*p[3]**2)-1*(xyVals[1]-xp[1])**2/(2*p[4]**2));
    return total;
    
def Mask2D(xyVals, p) :
    # Three 2D pucks with specified displacements. widths are assumed the same. for use with sVals:
    #   X border                        : p[0]
    #   Y border                         : p[1]

    x = xyVals[2];
    
    total = puck(xyVals, (x[0],p[0],x[1],p[1]));
    total = total + 1.0;
    total[total > 1.5] = 100000.0;
    return total;    
    
def Mask2DSG(xyVals, p) :
    # Three 2D pucks with specified displacements. widths are assumed the same. for use with sVals:
    #   X border                        : p[0]
    #   Y border                         : p[1]

    xm = xyVals[2];
    xz = xyVals[3];
    xp = xyVals[4];
    
    total = puck(xyVals, (xm[0],p[0],xm[1],p[1]));
    total = total + puck(xyVals, (xz[0],p[0],xm[1],p[1]));
    total = total + puck(xyVals, (xp[0],p[0],xp[1],p[1]));
    total = total + 1.0;
    total[total > 1.5] = 100000.0;
    return total;

def absline(xyVals, p) :
    # absolute value f a line
    #   intersept       : p[0]
    #   slope          : p[1]
    return numpy.abs(p[0] + p[1]*xyVals[0]);

def avdcrossing(xyVals, p) :
    # absolute value f a line
    #   intersept       : p[0]
    #   slope          : p[1]
    #   gap          : p[2]
    return numpy.sqrt((p[0] + p[1]*xyVals[0])**2 + p[2]**2.0);

    
#==============================================================================
# 
# Functions to execute the fit to IanStyleFits
#
#==============================================================================

def curve_fit_qgas(func, p_guess, zVals, xyVals, sVals = None, **kw):
    ''' curve_fit_qgas extends the operation of the scipy curve fit
        to more naturally deal with higher dimensional functions
        
        func : the function to be fit, formed as func(xyVals, p)
            xyVals:  is a tuple or list or array of arrays, each 
            xyVals[0] ... xyVals[N] is an array of coordinates
            so for example compare the N-dimensional function 
            evaulated at xyVals[0][q] ... xyVals[N][q] to zVals[q]
            each of the xyVals[p] can be a matrix, as they will 
            be .ravel()'ed to make 1D arrays internally.
            
            p : is the array of parameters
        
        p_guess : the initial guess of parameters
        
        zVals : data        
        
        xyVals : coordinates where data is known (as described in func)
        
        sVals : uncertanties on each point, defaults to 1
        
        kw : additional paramaters to pass to scipy.optimize.leastsq
    '''
 
    #==============
    # make sure that zVals, xyVals[], and svals are numpy arrays
    # If they are already,these functions will still make local copies
    # this may be slow, but in a fit the fit loop will be the problem
    #==============

    p_guessInt = numpy.array(p_guess, dtype=float);

    zValsInt = numpy.array(zVals, dtype=float);
    
    xyValsInt = [numpy.array(x, dtype=float) for x in xyVals];
    
    # If no uncertanties were passed set them to 1
    if (sVals is not None):
        sValsInt = numpy.array(sVals, dtype=float);
    else:
        sValsInt = None;
                
    # construct the desired fit function
    func =  FitFunctionForOptimize(func, zValsInt, xyValsInt, sVals = sValsInt);

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop('full_output', False)
    res = scipy.optimize.leastsq(func, p_guessInt, full_output=1, **kw)
    (popt, pcov, infodict, errmsg, ier) = res

    if ier not in [1,2,3,4]:
        msg = "Optimal parameters not found: " + errmsg
        #print msg 
        #return p_guessInt, p_guessInt  #just sends back random stuff to keep the program running
        raise RuntimeError(msg) # this aborts the batchrun program nicely

    # Generate covariance matrix
    if ( zValsInt.size > p_guessInt.size ) and pcov is not None:
        s_sq = (func(popt)**2).sum()/(zValsInt.size-p_guessInt.size);
        pcov = pcov * s_sq;
    else:
        pcov = numpy.inf;

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


# constructor function
def FitFunctionForOptimize(func, zVals, xyVals, sVals = None):
    '''
    Returns a function which takes "p" as a paramater for scipy.optimize
    func is the function to be fit of the form gaussian2D(xyVals, p), 
    xyVals = (xVals, yVals, ...)
    is a tuple of values where zVals are defined
    sVals is the array of uncertanties if it passed
    '''

    # define the internal function to return the sequence of residuals as a 1D array.
    
    # If no uncertanties were passed, proceed as if they are equal to 1
    if (sVals is None):
        def OptFunct(p):
            return ((func(xyVals, p) - zVals)).ravel();
    else:
        def OptFunct(p):
            return ((func(xyVals, p) - zVals)/sVals).ravel();
        
    
    return OptFunct;
    
#==============================================================================
# 
# Some generic functions
#
#==============================================================================

# Correct the optical depth for intensity and Doppler shift
# ffects.
    
def CorrectOD(ODRaw, CountsRaw, PulseTime, ISatCounts, tau):
    """
    Gives the corrected optical depth, given:
    ODRaw           The measured OD
    CountsRaw       Number of counts w/o atoms
    PulseTime       Imaging PulseDuration
    ISatCounts      ISat in count
    tau             recoil time (19 us for 40K, 42 us for 87Rb)    
    """
    
    IoverIsat=CountsRaw / ISatCounts;
    
    ODCorrect = -numpy.log(( IoverIsat * numpy.exp(-ODRaw) + 1.0)/(IoverIsat  + 1.0));
    ODCorrect += -( 1.0 / ( IoverIsat * numpy.exp(-ODRaw) + 1.0) - 1.0 / ( IoverIsat + 1.0) );
    ODCorrect *= (1.333)*(PulseTime/tau)**2;

    ODCorrect += ODRaw + IoverIsat * (1.0 - numpy.exp(-ODRaw));


    return ODCorrect;
    
    
