#%% Import 

import numpy as np

#%% Conductance fun

def ped(x1,x2):
    """
    Equation (3) in Robinson et al (1987)

    Input
    =====
    x1     : e- average energy [keV]
    x2     : e- energy flux    [ergs/cm², or equivalently mW/m²]

    Output
    ======
    Sigmap : Pedersen conductance [mho, or equiv. siemens, S]

    """
    Sigmap = 40*x1/(16+x1**2)*np.sqrt(x2)
    return Sigmap


def hall(x1,x2):
    """
    Equation (4) in Robinson et al (1987)

    Input
    =====
    x1     : e- average energy [keV]
    x2     : e- energy flux    [ergs/cm², or equivalently mW/m²]

    Output
    ======
    Sigmah : Hall conductance [mho, or equiv. siemens, S]
    """
    Sigmah = 18*x1**(1.85)/(16+x1**2)*np.sqrt(x2) 
    return Sigmah


def peduncertainty(x1,x2,dx1,dx2,varx1x2):
    """
    dSigmaP = peduncertainty(x1,x2,dx1,dx2,varx1x2)

    Calc uncertainty in Pedersen conductance given by Equation (3) in Robinson et al (1987)

    Input
    =====
    x1      : e- average energy                           [keV]
    x2      : e- energy flux                              [ergs/cm², or equivalently mW/m²]
    dx1     : Uncertainty/std deviation of e- avg energy  [keV]
    dx2     : Uncertainty/std deviation of e- energy flux [ergs/cm²]
    varx1x2 : Covariance of e- avg energy and energy flux [keV-ergs/cm²]

    Output
    ======
    dSigmap : Uncertainty in Sigmap [mho, or equiv. S, siemens]
    """

    if x2 == 0:
        # Conductance is proportional to sqrt(energy flux), so its derivative
        # is infinite at zero flux and linear error propagation cannot be used.
        # Report the one-sided conductance excursion from Fe=0 to Fe=dFe.
        # E0 uncertainty does not contribute at Fe=0 because conductance is
        # zero for every E0 there.
        dSigmap = ped(x1, dx2)
    else:
        # derivative of Sigmap wrt average energy
        denom = 16+x1**2
        dsp_dx1 = (40/denom - 80*(x1/denom)**2)*np.sqrt(x2)
        dsp_dx2 = 40*x1/denom/2/np.sqrt(x2)
        dSigmap = np.sqrt(dsp_dx1**2 * dx1**2 + dsp_dx2**2 * dx2**2 + 2 * dsp_dx1 * dsp_dx2 * varx1x2)

    return dSigmap

def halluncertainty(x1,x2,dx1,dx2,varx1x2):
    """
    dSigmaH = halluncertainty(x1,x2,dx1,dx2,varx1x2)

    Calc uncertainty in Hall conductance given by Equation (4) in Robinson et al (1987)

    Input
    =====
    x1      : e- average energy                           [keV]
    x2      : e- energy flux                              [ergs/cm², or equivalently mW/m²]
    dx1     : Uncertainty/std deviation of e- avg energy  [keV]
    dx2     : Uncertainty/std deviation of e- energy flux [ergs/cm²]
    varx1x2 : Covariance of e- avg energy and energy flux [keV-ergs/cm²]

    Output
    ======
    dSigmah : Uncertainty in Sigmah [mho, or equiv. S, siemens]
    """

    if x2 == 0:
        # Use the same one-sided Fe=dFe excursion as Pedersen conductance.
        dSigmah = hall(x1, dx2)
    else:
        # derivative of Sigmah wrt average energy
        denom = 16 + x1**2
        dsh_dx1 = 18 * x1**(0.85) / denom * (1.85 - 2 * x1**2 / denom) * np.sqrt(x2)
        dsh_dx2 =  9 * x1**(1.85) / denom / np.sqrt(x2)
        dSigmah = np.sqrt(dsh_dx1**2 * dx1**2 + dsh_dx2**2 * dx2**2 + 2 * dsh_dx1 * dsh_dx2 * varx1x2)

    return dSigmah
