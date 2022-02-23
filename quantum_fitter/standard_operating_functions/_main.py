import numpy as np
 
def calcRotationAngle(vComplex):
    """Following function copied from Labber to ensure things are consistent. Calculate angle that will rotate the signal in the input complex vector
    to the real component. To use on a array:
        angle = calcRotationAngle(Y)
        y_rotated = np.real(np.exp(1j*angle)*Y)
        
    Args:
        vComplex (array): complex vector array

    Returns:
        angle: angle that will rotate the signal in the input complex vector to the real component.
    """
    # remove nan's and infinites
    vComplex = vComplex[np.isfinite(vComplex)]
    # make sure we have data, check if complex
    if len(vComplex)<2:
        return 0.0
    if not np.any(np.iscomplex(vComplex)):
        return 0.0
    vPoly1 = np.polyfit(vComplex.real, vComplex.imag, 1)
    vPoly2 = np.polyfit(vComplex.imag, vComplex.real, 1)
    # get slope from smallest value of dy/dx and dx/dy
    if abs(vPoly1[0]) < abs(vPoly2[0]):
        angle = np.arctan(vPoly1[0])
    else:
        angle = np.pi/2.0 - np.arctan(vPoly2[0])
        if angle > np.pi:
            angle -= np.pi
    angle = -angle
    # try to make features appear as peaks instead of dips by adding pi
    data = np.real(vComplex * np.exp(1j*angle))
    meanValue = np.mean(data)
    # get metrics 
    first = abs(data[0] - meanValue)
    low = abs(np.min(data) - meanValue)
    high = abs(np.max(data) - meanValue)
    # method: use first point if first signal > 0.5 of max or min
    if first > 0.5*max(low,high):
        # approach 1: check first point (good for oscillations)
        if data[0] < meanValue:
            angle += np.pi
    else:
        # approach 2: check max/min points
        if high < low:
            angle += np.pi
#    # approach 2: check mean vs median (good for peaks)
#    if meanValue < np.median(data):
#        angle += np.pi
    return angle
