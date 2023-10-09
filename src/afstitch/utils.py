"""


"""
import numpy as np
from scipy.signal import fftconvolve

# Polynomial coefficients for the autocorrelation of the triangular function
C = np.array([[1., -3., 3., -1],
              [4., 0, -6., 3.],
              [1., 3., 3., -3.],
              [0., 0., 0., 1.]])

# Conversion matrix
M = np.array([np.multiply.outer(C[j], C[i]).ravel() for i in range(4) for j in range(4)]).T

# Exponent arrays for 0th, 1st and 2nd derivatives
E0 = np.array([0,1,2,3])
E1 = np.array([0,0,1,2])
E2 = np.array([0,0,0,1])

def reshape_to_view(a, shifts, shape):
    """

    Reshaping the dimensions of image "a" to fit tightly the total viewport 
    given by the list of "shifts" and target view "shape".

    :param a: Input image
    :type a: 2d array
    :param shifts: Shifts in pixel coordinates
    :type shifts: list[tuple] or 2d array
    :param shape: Target shape
    :type shape: tuple
    :return: Reshaped image, Adjusted shifts, Offset  
    :rtype: list[ndarray]

    """

    # Calculate new shifts and offset
    shifts = np.asarray(shifts, dtype=int)
    min0, min1 = shifts.min(axis=0)
    max0, max1 = shifts.max(axis=0) + shape
    shifts_new = shifts - (min0, min1)
    offset_new = np.array([min0, min1])

    # Calculate target shape
    ash = a.shape
    nash = (max0 - min0, max1 - min1)

    # Target shape equals input shape
    if ash == nash:
        return a.copy(), shifts_new, offset_new

    # Create new reshaped image
    a_new = np.zeros(nash, dtype=a.dtype)

    # Calculate limits along first dimension
    s0 = max(min0, 0)
    new_s0 = max(-min0, 0)
    e0 = min(ash[0], nash[0] + min0)
    new_e0 = min(nash[0], ash[0] - min0)

    # Calculate limits along second dimension
    s1 = max(min1, 0)
    new_s1 = max(-min1, 0)
    e1 = min(ash[1], nash[1] + min1)
    new_e1 = min(nash[1], ash[1] - min1)

    # Copy image values
    a_new[new_s0:new_e0, new_s1:new_e1] = a[s0:e0, s1:e1]

    return a_new, shifts_new, offset_new


def cc(A, B, mode='full'):
    """

    A fast cross-correlation based on scipy.signal.fftconvolve.

    :param A: The reference image
    :param B: The template image to match
    :param mode: one of 'same', 'full' (default) or 'valid' (see help for fftconvolve for more info)
    :return: The cross-correlation of 'A' and 'B'.
    """
    return fftconvolve(A, B[::-1, ::-1], mode=mode)


def allf(x0, c):
    """

    Interpolant and derivatives at 'x0'.

    :param x0: ??
    :type x0: 1d array of shape (2,)
    :param c: Generator vector
    :return: ??
    :rtype: ndarray

    """
    x,y = x0
    return np.dot(np.array([np.multiply.outer(        y**E0,         x**E0).ravel(),
                            np.multiply.outer(        y**E0,    E0 * x**E1).ravel(),
                            np.multiply.outer(   E0 * y**E1,         x**E0).ravel(),
                            np.multiply.outer(        y**E0, E1*E0 * x**E2).ravel(),
                            np.multiply.outer(   E0 * y**E1,    E0 * x**E1).ravel(),
                            np.multiply.outer(E0*E1 * y**E2,         x**E0).ravel()]), c)

def sub_pix_min(a):
    """

    Find the position of the minimum in 2D array 'a' with subpixel precision.

    :param a: Input array
    :type a: 2d array
    :return: Position of minimum in 'a'
    :rtype: float

    """
    sh = a.shape
    
    # Find the global minimum
    cmin = np.array(np.unravel_index(a.argmin(), sh))

    # Move away from edges
    if cmin[0] < 2:
        cmin[0] = 2
    elif cmin[0]+2 >= sh[0]:
        cmin[0] = sh[0] - 3
    if cmin[1] < 2:
        cmin[1] = 2
    elif cmin[1]+2 >= sh[1]:
        cmin[1] = sh[1] - 3

    # Sub-pixel minimum position.
    a0 = a[(cmin[0]-2):(cmin[0]+3), (cmin[1]-2):(cmin[1]+3)]
  
    # Find which quandrant to refine
    jp = 1 if a0[2,3]<a0[2,1] else 0
    ip = 1 if a0[3,2]<a0[1,2] else 0
    af = a0[ip:ip+4, jp:jp+4]

    # Generate vector
    c = np.dot(M, af.ravel())

    # Newton-Raphson
    x0 = np.array([1.-ip, 1.-jp])
    tol = 1e-8
    for i in range(15):
        f, fx, fy, fxx, fxy, fyy = allf(x0, c)
        dx = -np.array([fyy*fx - fxy*fy, -fxy*fx + fxx*fy])/(fxx*fyy - fxy*fxy)
        x0 += dx
        if (dx*dx).sum() < tol:
            break
    r = x0 - 1 + np.array([ip, jp]) + cmin

    return r

def match(Ib, It, Ifact=None, mb=None, mt=None, scale=True, max_shift=None):
    """

    Solve the generalized matching problem

    L(s) = 1/N(s) * sum_r mb[r-s] * mt[r] * (Ifact[r] * Ib[r - s] - It[r])**2

    with N(s) = sum_r mb[r-s] * mt[r]

    :param Ib: Base image on which to match template
    :param It: Template image
    :param Ifact: Multiplier
    :param mb: Base image mask
    :param mt: Template mask
    :param scale: True, False or 'phase"
    :param max_shift: maximum relative shift allowed. [TODO: might not need to be around 0]

    :return: A dict containing:
      * 'r', the two-dimensional vector shift
      * 'scale', the optimal scaling factor
      * 'D', the value of the optimization function at the minimum
      * 'status', a code and message telling if everything went ok.
      * maybe something else in the future

    """
    # Check dimensions
    rsh = Ib.shape
    tsh = It.shape
    if (tsh[0] > rsh[0]) or (tsh[1] > rsh[1]):
        raise RuntimeError('The base image should be larger than the template.')

    # Center location
    ctr = (np.array(rsh) + np.array(tsh)) // 2

    # Create template mask
    if mt is None:
        mt = np.ones_like(It, dtype=float)

    # Compute elements of distance metric depending on existence of input arrays.
    if mb is None:
        N = mt.sum()
        T2 = (mt*abs(It)**2).sum()
        if Ifact is None:
            T1 = cc(abs(Ib)**2, mt)
            T3 = cc(Ib, mt*It.conj())
        else:
            T1 = cc(abs(Ib)**2, mt*abs(Ifact)**2)
            T3 = cc(Ib, mt*Ifact*It.conj())
    else:
        N = cc(mb, mt)
        T2 = cc(mb, mt*abs(It)**2)
        if Ifact is None:
            T1 = cc(mb*abs(Ib)**2, mt)
            T3 = cc(mb*Ib, mt*It.conj())
        else:
            T1 = cc(mb*abs(Ib)**2, mt*abs(Ifact)**2)
            T3 = cc(mb*Ib, mt*Ifact*It.conj())

    # Compute scaling factor if required
    c = 1.
    if scale is not False:
        c = T3/T2
        if scale == 'phase':
            c /= abs(c)

    # Construct distance map
    D = (T1 + T2*abs(c)**2 - 2*np.real(np.conj(c)*T3))/(N + 1e-8)

    if max_shift is not None:
        ii0, ii1 = np.ogrid[:D.shape[0], :D.shape[1]]
        c = (((ii0 - tsh[0] - 1)**2 + (ii1 - tsh[1] - 1)**2) > max_shift**2)
        D[c] = D.max()

    # Find subpixel optimum
    r = sub_pix_min(D)
    if scale is not False:
        s = c[tuple((np.round(r).astype(int)))]
    else:
        s = 1.
    r -= tsh
    r += 1

    return {'r': r, 'scale': s, 'D': D, 'status': 'OK'}
