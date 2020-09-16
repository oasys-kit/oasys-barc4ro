#!/usr/bin/python
# coding: utf-8

###################################################################################
# barc4RefractiveOptics
# Authors/Contributors: Rafael Celestre, Oleg Chubar, Manuel Sanchez del Rio
# Rafael.Celestre@esrf.eu
# creation: 24.06.2019
# last update: 30.07.2020 (v0.4)
#
# Check documentation:
# Celestre, R. et al. (2020). Recent developments in x-ray lenses modelling with SRW.
# Proc. SPIE 11493, Advances in Computational Methods for X-Ray Optics V, 11493-17.
# arXiv:2007.15461 [physics.optics] - https://arxiv.org/abs/2007.15461
#
# Copyright (c) 2019-2020 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

import numpy as np
import warnings

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Wavefront fitting routines
# ----------------------------------------------------------------------------------------------------------------------

'''
This library contains pieces of Python code available on several GitHub repositories from other authors. This library
part of fixes bugs, corrects mistakes and brings the parts of those codes to Python 3.x as well as implements new
features. The functions from other authors are renamed to provide more uniformity and clarity when using this library.

----------------------------------------------
 Wavefront fitting
----------------------------------------------

The three sets of orthonormal polynomials for fitting wavefronts are:

a) Zernike circle polynomials
ref[1] Virendra N. Mahajan,
       "Zernike Circle Polynomials and Optical Aberrations of Systems with Circular Pupils,"
       Appl. Opt. 33, 8121-8124 (1994)

b) Rectangular polynomials
ref[2] Virendra N. Mahajan and Guang-ming Dai,
       "Orthonormal polynomials in wavefront analysis: analytical solution,"
       J. Opt. Soc. Am. A 24, 2994-3016 (2007)

ref[3] Virendra N. Mahajan,
       "Orthonormal polynomials in wavefront analysis: analytical solution: errata,"
       J. Opt. Soc. Am. A 29, 1673-1674 (2012)

c) 2D Legendre polynomials
ref[4] Virendra N. Mahajan,
       "Orthonormal aberration polynomials for anamorphic optical imaging systems with circular pupils,"
       Appl. Opt. 51, 4087-4091 (2012)

Please, check also:

ref[5] Jingfei Ye, Zhishan Gao, Shuai Wang, Jinlong Cheng, Wei Wang, and Wenqing Sun,
       "Comparative assessment of orthogonal polynomials for wavefront reconstruction over the square aperture,"
       J. Opt. Soc. Am. A 31, 2304-2311 (2014)

Rafael Celestre would like the following people for discussions and for pointing me to relevant literature:
    > Prof. Virendra Mahajan - University of Arizona, USA
    > Prof. Herbert Gross - University of Jena, Germany

----------------------------------------------
Python routines:
----------------------------------------------
===> Zernike circle polynomials
Imported and adapted from "libtim-py: miscellaneous data manipulation utils"
(github.com/tvwerkhoven/libtim-py)
Module: uti_optics.py
Created by: Tim van Werkhoven (tim@vanwerkhoven.org)
Converted from Python 2.7 to 3.6 by Manuel Sanchez del Rio - ESRF
Adapted by: Rafael Celestre - ESRF (31.05.2018)

===> Rectangular aperture Zernike base
The coefficients in 'zernike_rec()' were imported from "opticspy: python optics mode"
(https://github.com/Sterncat/opticspy)
Module: zernike_rec.py
Created by: Xing Fan (marvin.fanxing@gmail.com) based on the equations from ref[2]. Corrections from ref[3] implemented 
by Rafael Celestre - ESRF (27.02.2020)
Functions written to mimic the ones from 'libtim-py': Rafael Celestre - ESRF (27.02.2020)

===> Legendre Pol. base
Functions written to mimic the ones from 'libtim-py': Rafael Celestre - ESRF (29.02.2020)

'''

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Zernike circle polynomials
# ----------------------------------------------------------------------------------------------------------------------

def calc_zern_circ_basis(nmodes, rad, modestart=1, calc_covmat=False):
    """
    Calculate a basis of **nmodes** Zernike modes with radius **rad**.

    ((If **mask** is true, set everything outside of radius **rad** to zero (default). If this is not done, the set of
    Zernikes will be **rad** by **rad** square and are not orthogonal anymore.)) --> Nothing is masked, do this manually
    using the 'mask' entry in the returned dict.

    This output of this function can be used as cache for other functions.

    @param [in] nmodes Number of modes to generate
    @param [in] rad Radius of Zernike modes
    @param [in] modestart First mode to calculate (Noll index, i.e. 1=piston)
    @param [in] calc_covmat Return covariance matrix for Zernike modes, and its inverse
    @return Dict with entries 'modes' a list of Zernike modes, 'modesmat' a matrix of (nmodes, npixels), 'covmat' a
    covariance matrix for all these modes with 'covmat_in' its inverse, 'mask' is a binary mask to crop only the
    orthogonal part of the modes.
    """

    if (nmodes <= 0):
        return {'modes':[], 'modesmat':[], 'covmat':0, 'covmat_in':0, 'mask':[[0]]}
    if (rad <= 0):
        raise ValueError("radius should be > 0")
    if (modestart <= 0):
        raise ValueError("**modestart** Noll index should be > 0")

    # Use vectors instead of a grid matrix
    rvec = ((np.arange(2.0*rad) - rad)/rad)
    r0 = rvec.reshape(-1,1)
    r1 = rvec.reshape(1,-1)

    grid_rad = mk_circ_mask(2*rad)
    grid_ang = np.arctan2(r0, r1)

    grid_mask = grid_rad <= 1

    # Build list of Zernike modes, these are *not* masked/cropped
    zern_modes = [zernike_circ(zmode, grid_rad, grid_ang) for zmode in range(modestart, nmodes+modestart)]

    # Convert modes to (nmodes, npixels) matrix
    zern_modes_mat = np.r_[zern_modes].reshape(nmodes, -1)

    covmat = covmat_in = None
    if (calc_covmat):
        # Calculate covariance matrix
        covmat = np.array([[np.sum(zerni * zernj * grid_mask) for zerni in zern_modes] for zernj in zern_modes])
        # Invert covariance matrix using SVD
        covmat_in = np.linalg.pinv(covmat)

    # Create and return dict
    return {'modes': zern_modes, 'modesmat': zern_modes_mat, 'covmat':covmat, 'covmat_in':covmat_in, 'mask': grid_mask}


def fit_zernike_circ(wavefront, zern_data={}, nmodes=37, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                rec_zern=True, err=None):
    """
    Fit **nmodes** Zernike modes to a **wavefront**. The **wavefront** will be fit to Zernike modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Zernike mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **zern_data** as cache. If this is not given, it will be generated. See calc_zern_circ_basis()
    for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] zern_data Zernike basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Zernike modes to fit
    @param [in] rad Radius of Zernike modes to fit
    @param [in] rec_zern Reconstruct Zernike modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Zernike mode amplitudes,
    the second element is a full 2D Zernike reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_zern_circ_basis() for details on **zern_data** cache
    """

    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5*max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape)-rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode	< 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        rad = -rad * min(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * min(wavefront.shape)

    # Make cropping slices to select only central part of the wavefront
    xslice = slice(int(center[0]-rad), int(center[0]+rad))
    yslice = slice(int(center[1]-rad), int(center[1]+rad))

    # Compute Zernike basis if absent
    # if (not (zern_data.has_key('modes')):
    if (not ( 'modes' in zern_data.keys())):
        tmp_zern = calc_zern_circ_basis(nmodes, rad)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
        zern_data['modes'][0].shape != (2*rad, 2*rad)):
        tmp_zern = calc_zern_circ_basis(nmodes, rad)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
        # wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec] * weight)).ravel()
        # This is 5x faster:
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask

        # Crop out central region of wavefront, then only select the orthogonal part of the Zernike modes (grid_mask)
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
        # wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec])).ravel()
        # This is 5x faster
        # RC190225:
        # FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M
        # and N are the input matrix dimensions. To use the future default and silence this warning we advise to pass
        # `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
        # wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel())[0]
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(),rcond=None)[0]

    wf_zern_vec[:startmode-1] = 0

    # Calculate full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_circ(wf_zern_vec, zern_data=zern_data, rad=min(wavefront.shape)/2)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid**2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean()**2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)


def calc_zernike_circ(zern_vec, rad, zern_data={}, mask=True):
    """
    Constructs wavefront with Zernike amplitudes **zern_vec**. Given vector **zern_vec** with the amplitude of Zernike
    modes, return the reconstructed wavefront with radius **rad**. This function uses **zern_data** as cache. If this is
    not given, it will be generated. See calc_zern_circ_basis() for details. If **mask** is True, set everything outside
    radius **rad** to zero, this is the default and will use orthogonal Zernikes. If this is False, the modes will not
    be cropped.

    @param [in] zern_vec 1D vector of Zernike amplitudes
    @param [in] rad Radius for Zernike modes to construct
    @param [in] zern_data Zernike basis cache
    @param [in] mask If True, set everything outside the Zernike aperture to zero, otherwise leave as is.
    @see See calc_zern_circ_basis() for details on **zern_data** cache and **mask**
    """

    from functools import reduce

    if (not ('modes' in zern_data.keys()) ):
        tmp_zern = calc_zern_circ_basis(len(zern_vec), rad)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (len(zern_vec) > len(zern_data['modes'])):
        tmp_zern = calc_zern_circ_basis(len(zern_vec), rad)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    zern_basis = zern_data['modes']

    gridmask = 1
    if (mask):
        gridmask = zern_data['mask']

    # Reconstruct the wavefront by summing modes
    return reduce(lambda x,y: x+y[1]*zern_basis[y[0]] * gridmask, enumerate(zern_vec), 0)


def zernike_circ(j, rho, phi, norm=True):
    """
    Calculates Zernike mode with Noll-index j on a square grid of <size>^2
    elements
    """
    n, m = noll_to_zern(j)
    return zernike_circ_mn(m, n, rho, phi, norm)


def zernike_circ_mn(m, n, rho, phi, norm=True):
    """
    Calculates Zernike mode (m,n) on grid **rho** and **phi**.

    **rho** and **phi** should be radial and azimuthal coordinate grids of identical shape, respectively.

    @param [in] m Radial Zernike index
    @param [in] n Azimuthal Zernike index
    @param [in] rho Radial coordinate grid
    @param [in] phi Azimuthal coordinate grid
    @param [in] norm Normalize modes to unit variance
    @return Zernike mode (m,n) with identical shape as rho, phi
    @see <http://research.opt.indiana.edu/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html> and
    <http://research.opt.indiana.edu/Library/HVO/Handbook.html>.
    """
    nc = 1.0
    if (norm):
        nc = (2*(n+1)/(1+(m==0)))**0.5
    if (m > 0): return nc*zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return nc*zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return nc*zernike_rad(0, n, rho)


def zernike_rad(m, n, rho):
    """
    Make radial Zernike polynomial on 1D - coordinate grid **rho**.

    @param [in] m Radial Zernike index
    @param [in] n Azimuthal Zernike index
    @param [in] rho Radial coordinate grid
    @return Radial polynomial with identical shape as **rho**
    """


    from scipy.special import factorial as fac

    if (np.mod(n-m, 2) == 1):
        return rho*0.0

    wf = rho*0.0
    for k in range(int((n-m)/2+1)):
        wf += rho**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )

    return wf


def noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.

    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.

    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)


def zern_normalisation(nmodes=37):
    """
    Calculates normalisation vector.

    This function calculates a **nmodes** element vector with normalisation constants for Zernike modes that have not
    already been normalised.

    @param [in] nmodes Size of normalisation vector.
    @see <http://research.opt.indiana.edu/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html> and
    <http://research.opt.indiana.edu/Library/HVO/Handbook.html>.
    """

    nolls = (noll_to_zern(j+1) for j in range(nmodes))
    norms = [(2*(n+1)/(1+(m==0)))**0.5 for n, m  in nolls]
    return np.asanyarray(norms)


def mk_circ_mask(r0, r1=None, norm=True, center=None, dtype=np.float, getxy=False):
    """
    Make a rectangular matrix of size (r0, r1) where the value of each element is the Euclidean distance to **center**.
    If **center** is not given, it is the middle of the matrix. If **norm** is True (default), the distance is
    normalized to half the radius, i.e. values will range from [-1, 1] for both axes. If only r0 is given, the matrix
    will be (r0, r0). If r1 is also given, the matrix will be (r0, r1). To make a circular binary mask of (r0, r0), use
    mk_circ_mask(r0) < 1

    @param [in] r0 The width (and height if r1==None) of the mask.
    @param [in] r1 The height of the mask.
    @param [in] norm Normalize the distance such that 2/(r0, r1) equals a distance of 1.
    @param [in] getxy Return x, y-values instead of r
    @param [in] dtype Datatype to use for radial coordinates
    @param [in] center Set distance origin to **center** (defaults to the middle of the rectangle)
    """

    if (not r1):
        r1 = r0
    if (r0 < 0 or r1 < 0):
        warnings.warn("mk_circ_mask(): r0 < 0 or r1 < 0?")

    if (center != None and norm and sum(center)/len(center) > 1):
        raise ValueError("|center| should be < 1 if norm is set")

    if (center == None):
        if (norm): center = (0, 0)
        else: center = (r0/2.0, r1/2.0)

    if (norm):
        r0v = np.linspace(-1-center[0], 1-center[0], int(r0)).astype(dtype).reshape(-1,1)
        r1v = np.linspace(-1-center[1], 1-center[1], int(r1)).astype(dtype).reshape(1,-1)
    else:
        r0v = np.linspace(0-center[0], r0-center[0], int(r0)).astype(dtype).reshape(-1,1)
        r1v = np.linspace(0-center[1], r1-center[1], int(r1)).astype(dtype).reshape(1,-1)

    if (getxy):
        return r0v, r1v
    else:
        return (r0v**2. + r1v**2.)**0.5

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Rectangular aperture Zernike base
# ----------------------------------------------------------------------------------------------------------------------

def calc_zern_rec_basis(nmodes, npix, modestart=1, calc_covmat=False):
    """
    Calculates a basis of **nmodes** Zernike modes with normalised dimensions [2*np.sqrt(1-a**2), 2a].

    This output of this function can be used as cache for other functions.

    @param [in] nmodes Number of modes to generate
    @param [in] npix array size to calculate the basis
    @param [in] modestart First mode to calculate (Noll index, i.e. 1=piston)
    @param [in] calc_covmat Return covariance matrix for Zernike modes, and its inverse
    @return Dict with entries 'modes' a list of Zernike modes, 'modesmat' a matrix of (nmodes, npixels), 'covmat' a
    covariance matrix for all these modes with 'covmat_in' its inverse, 'mask' is a binary mask to crop only the
    orthogonal part of the modes.
    """

    if (nmodes <= 0):
        return {'modes':[], 'modesmat':[], 'covmat':0, 'covmat_in':0, 'mask':[[0]]}
    if (len(npix) is not 2):
        raise ValueError("npix sould be [ny, nx]")
    if (modestart <= 0):
        raise ValueError("**modestart** Noll index should be > 0")
    if (modestart > 15):
        raise ValueError("**modestart** Noll index should be < 15")

    a = npix[1]/2/np.sqrt((npix[0]/2)**2 + (npix[1]/2)**2)
    x_axis = np.linspace(-a, a, npix[1])
    y_axis = np.linspace(-np.sqrt(1-a**2), np.sqrt(1-a**2), npix[0])
    X, Y = np.meshgrid(x_axis, y_axis)
    grid_rad = np.sqrt(X ** 2 + Y ** 2)
    grid_ang = np.arctan2(Y, X)
    grid_mask = grid_rad >= 0

    # Build list of Zernike modes, these are *not* masked/cropped
    zern_modes = [zernike_rec(zmode, a, grid_rad, grid_ang) for zmode in range(modestart, nmodes+modestart)]

    # Convert modes to (nmodes, npixels) matrix
    zern_modes_mat = np.r_[zern_modes].reshape(nmodes, -1)

    covmat = covmat_in = None
    if (calc_covmat):
        # Calculates covariance matrix
        covmat = np.array([[np.sum(zerni * zernj) for zerni in zern_modes] for zernj in zern_modes])
        # Invert covariance matrix using SVD
        covmat_in = np.linalg.pinv(covmat)

    # Create and return dict
    return {'modes': zern_modes, 'modesmat': zern_modes_mat, 'covmat':covmat, 'covmat_in':covmat_in, 'mask': grid_mask}


def fit_zernike_rec(wavefront, zern_data={}, nmodes=15, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                    rec_zern=True, err=None):
    """
    Fit **nmodes** Zernike modes to a **wavefront**. The **wavefront** will be fit to Zernike modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Zernike mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **zern_data** as cache. If this is not given, it will be generated.
    See calc_zern_circ_basis() for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] zern_data Zernike basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Zernike modes to fit
    @param [in] rad Radius of Zernike modes to fit
    @param [in] rec_zern Reconstruct Zernike modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Zernike mode amplitudes,
    the second element is a full 2D Zernike reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_zern_circ_basis() for details on **zern_data** cache
    """

    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5*max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape)-rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode	< 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        npix = -rad * 2 * np.array(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * wavefront.shape

    # Convert rad and center if coordinates are fractional
    xslice = slice(int(center[1]-npix[1]/2), int(center[1]+npix[1]/2))
    yslice = slice(int(center[0]-npix[0]/2), int(center[0]+npix[0]/2))

    # Compute Zernike basis if absent
    if (not ( 'modes' in zern_data.keys())):
        tmp_zern = calc_zern_rec_basis(nmodes, npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
        zern_data['modes'][0].shape != (npix[0], npix[1])):
        tmp_zern = calc_zern_rec_basis(nmodes, npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(),rcond=None)[0]

    wf_zern_vec[:startmode-1] = 0

    # Calculates full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_rec(wf_zern_vec, zern_data=zern_data, npix=wavefront.shape)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid**2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean()**2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)


def calc_zernike_rec(zern_vec, npix, zern_data={}, mask=True):
    """
    Constructs wavefront with Zernike amplitudes **zern_vec**. Given vector **zern_vec** with the amplitude of Zernike
    modes, return the reconstructed wavefront with radius **rad**. This function uses **zern_data** as cache. If this is
    not given, it will be generated. See calc_zern_circ_basis() for details. If **mask** is True, set everything outside
    radius **rad** to zero, this is the default and will use orthogonal Zernikes. If this is False, the modes will not
    be cropped.

    @param [in] zern_vec 1D vector of Zernike amplitudes
    @param [in] npix array size to calculate the basis
    @param [in] zern_data Zernike basis cache
    @param [in] mask If True, set everything outside the Zernike aperture to zero, otherwise leave as is.
    @see See calc_zern_rec_basis() for details on **zern_data** cache and **mask**
    """

    from functools import reduce

    if (not ('modes' in zern_data.keys()) ):
        tmp_zern = calc_zern_rec_basis(len(zern_vec), npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (len(zern_vec) > len(zern_data['modes'])):
        tmp_zern = calc_zern_rec_basis(len(zern_vec), npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    zern_basis = zern_data['modes']

    gridmask = 1
    if (mask):
        gridmask = zern_data['mask']

    # Reconstruct the wavefront by summing modes
    return reduce(lambda x,y: x+y[1]*zern_basis[y[0]], enumerate(zern_vec), 0)


def zernike_rec(j, a, rho, phi):
    """
    Calculates the orthonormal rectangular Zernike mode j on grid **rho** and **phi**.

    **rho** and **phi** should be radial and azimuthal coordinate grids of identical shape, respectively.

    @param [in] j Radial Zernike index
    @param [in] a unit rectangle inscribed a unit circle. Its corner points lie at a distance of unity from its centre.
    @param [in] rho Radial coordinate grid
    @param [in] phi Azimuthal coordinate grid
    @return Zernike mode j with identical shape as rho, phi

    """
    # TODO: check polynomials coefficients
    mu = np.sqrt(9 - 36*a**2 + 103*a**4 - 134*a**6 + 67*a**6 + 67*a**8)
    v = np.sqrt(49 - 196*a**2 + 330*a**4 - 268*a**6 + 134*a**8)
    tau = 1/(128*v*a**4*(1-a**2)**2)
    eta = 9 - 45*a**2 + 139*a**4 - 237*a**6 + 201*a**8 - 67*a**10
    chi = 35-70*a**2+62*a**4
    if j is 1:
        return np.ones(rho.shape)
    elif j is 2:
        return np.sqrt(3)/a*rho*np.cos(phi)
    elif j is 3:
        return np.sqrt(3/(1-a**2))*rho*np.sin(phi)
    elif j is 4:
        return np.sqrt(5)/2/np.sqrt(1-2*a**2+2*a**4)*(3*rho**2-1)
    elif j is 5:
        return 3/2/a/np.sqrt(1-a**2)*rho**2*np.sin(2*phi)
    elif j is 6:
        return np.sqrt(5)/2/a**2/(1-a**2)/np.sqrt(1-2*a**2+2*a**4)*(3*(1-2*a**2+2*a**4)*rho**2*np.cos(2*phi) + 3*(1-2*a**2)*rho**2-2*a**2*(1-a**2)*(1-2*a**2))
    elif j is 7:
        return np.sqrt(21)/2/np.sqrt(27-81*a**2+116*a**4-62*a**6)*(15*rho**2-9+4*a**2)*rho*np.sin(phi)
    elif j is 8:
        return np.sqrt(21)/2/a/np.sqrt(35-70*a**2+62*a**4)*(15*rho**2-5-4*a**2)*rho*np.cos(phi)
    elif j is 9:
        return (np.sqrt(5)*np.sqrt((27-54*a**2+62*a**4)/(1-a**2))/(8*a**2*(27-81*a**2+116*a**4-62*a**6)))*((27-54*a**2+62*a**4)*rho*np.sin(3*phi)-3*(4*a**2*(3-13*a**2+10*a**4)-(9-18*a**2-26*a**4))*rho*np.sin(phi))
    elif j is 10:
        return (np.sqrt(5)/(8*a**3*(1-a**2)*np.sqrt(chi)))*(chi*rho**3*np.cos(3*phi)-3*(4*a**2*(7-17*a**2+10*a**4) - chi*rho**2)*rho*np.cos(phi))
    elif j is 11:
        return 1/8/mu*(315*rho**4+30*(1-2*a**2)*rho**2*np.cos(2*phi)-240*rho**2+27+16*a*2-16*a**4)
    elif j is 12:
        return (3*mu/(8*a**2*v*eta))*(315*(1-2*a**2)*(1-2*a**2+2*a**4)*rho**4+5*(7*mu**2*rho**2-21+72*a**2-225*a**4 + 306*a**6-152*a**8)*rho**2*np.cos(2*phi)-15*(1-2*a**2)*(7+4*a**2-71*a**4+134*a**6-67*a**8)*rho**2 + a**2*(1-a**2)*(1-2*a**2)*(70-233*a**2+233*a**4))
    elif j is 13:
        return np.sqrt(21)/(4*a*np.sqrt(1-3*a**2+4*a**4-2*a**6))*(5*rho**2-3)*rho**2*np.sin(2*phi)
    elif j is 14:
        return 6*tau*(5*v**2*rho**4*np.cos(4*phi)-20*(1-2*a**2)*(6*a**2*(7-16*a**2+18*a**4-9*a**6) - 49*(1-2*a**2+2*a**4)*rho**2)*rho**2*np.cos(phi)+8*a**4*(1-a**2)**2*(21-62*a**2+62*a**4) - 120*a**2*(7-30*a**2+46*a**4-23*a**6)*rho**2+15*(49-196*a**2+282*a**4-172*a**6+86*a**8)*rho**4)
    elif j is 15:
        return (np.sqrt(21)/(8*a**3*np.sqrt((1-a**2)**3))/np.sqrt(1-2*a**2+2*a**4)) * (-(1-2*a**2)*(6*a**2-6*a**4-5*rho**2)*rho**2*np.sin(2*phi)+(5/2)*(1-2*a**2+2**a**4)*rho**4*np.sin(4*phi))


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Legendre Pol. base
# ----------------------------------------------------------------------------------------------------------------------

def calc_legendre_basis(nmodes, npix, modestart=1, calc_covmat=False):
    """
    Calculates a basis of **nmodes** Legendre modes with normalised dimensions [-1, 1].

    This output of this function can be used as cache for other functions.

    @param [in] nmodes Number of modes to generate
    @param [in] npix array size to calculate the basis
    @param [in] modestart First mode to calculate (Noll index, i.e. 1=piston)
    @param [in] calc_covmat Return covariance matrix for Legendre modes, and its inverse
    @return Dict with entries 'modes' a list of Legendre modes, 'modesmat' a matrix of (nmodes, npixels), 'covmat' a
    covariance matrix for all these modes with 'covmat_in' its inverse, 'mask' is a binary mask to crop only the
    orthogonal part of the modes.

    """

    if (nmodes <= 0):
        return {'modes':[], 'modesmat':[], 'covmat':0, 'covmat_in':0, 'mask':[[0]]}
    if (len(npix) is not 2):
        raise ValueError("npix sould be [ny, nx]")
    if (modestart <= 0):
        raise ValueError("**modestart** Noll index should be > 0")
    if (modestart > 44):
        raise ValueError("**modestart** Noll index should be < 45")

    x_axis = np.linspace(-1, 1, npix[1])
    y_axis = np.linspace(-1, 1, npix[0])
    X, Y = np.meshgrid(x_axis, y_axis)
    grid_mask = X >= -1

    # Build list of Legendre modes, these are *not* masked/cropped
    zern_modes = [legendre_2D(zmode, X, Y) for zmode in range(modestart, nmodes+modestart)]

    # Convert modes to (nmodes, npixels) matrix
    zern_modes_mat = np.r_[zern_modes].reshape(nmodes, -1)

    covmat = covmat_in = None
    if (calc_covmat):
        # Calculates covariance matrix
        covmat = np.array([[np.sum(zerni * zernj) for zerni in zern_modes] for zernj in zern_modes])
        # Invert covariance matrix using SVD
        covmat_in = np.linalg.pinv(covmat)

    # Create and return dict
    return {'modes': zern_modes, 'modesmat': zern_modes_mat, 'covmat':covmat, 'covmat_in':covmat_in, 'mask': grid_mask}


def fit_legendre(wavefront, leg_data={}, nmodes=44, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5,
                    rec_leg=True, err=None):

    """
    Fit **nmodes** Legendre modes to a **wavefront**. The **wavefront** will be fit to Legendre modes for a circle with
    radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes. If **center**
    or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape. **startmode**
    indicates the Legendre mode (Noll index) to start fitting with, i.e. ***startmode** = 4 will skip piston, tip and
    tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned
    vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as
    these can sometimes not be derived from data. If **err** is an empty list, it will be filled with measures for the
    fitting error:
    1. Mean squared difference
    2. Mean absolute difference
    3. Mean absolute difference squared

    This function uses **leg_data** as cache. If this is not given, it will be generated.
    See calc_legendre_basis() for details.

    @param [in] wavefront Input wavefront to fit
    @param [in] leg_data Legendre basis cache
    @param [in] nmodes Number of modes to fit
    @param [in] startmode Start fitting at this mode (Noll index)
    @param [in] fitweight Mask to use as weights when fitting
    @param [in] center Center of Legendre modes to fit
    @param [in] rad Radius of Legendre modes to fit
    @param [in] rec_leg Reconstruct Legendre modes and calculate errors.
    @param [out] err Fitting errors
    @return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Legendre mode amplitudes,
    the second element is a full 2D Legendre reconstruction and the last element is the 2D difference between the input
    wavefront and the full reconstruction.
    @see See calc_legendre_basis() for details on **leg_data** cache
    """
    zern_data = leg_data
    rec_zern = rec_leg
    if (rad < -1 or min(center) < -1):
        raise ValueError("illegal radius or center < -1")
    elif (rad > 0.5*max(wavefront.shape)):
        raise ValueError("radius exceeds wavefront shape?")
    elif (max(center) > max(wavefront.shape)-rad):
        raise ValueError("fitmask shape exceeds wavefront shape?")
    elif (startmode	< 1):
        raise ValueError("startmode<1 is not a valid Noll index")

    # Convert rad and center if coordinates are fractional
    if (rad < 0):
        npix = -rad * 2 * np.array(wavefront.shape)
    if (min(center) < 0):
        center = -np.r_[center] * wavefront.shape

    # Convert rad and center if coordinates are fractional
    xslice = slice(int(center[1]-npix[1]/2), int(center[1]+npix[1]/2))
    yslice = slice(int(center[0]-npix[0]/2), int(center[0]+npix[0]/2))

    # Compute Zernike basis if absent
    if (not ( 'modes' in zern_data.keys())):
        tmp_zern = calc_legendre_basis(nmodes, npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (nmodes > len(zern_data['modes']) or
        zern_data['modes'][0].shape != (npix[0], npix[1])):
        tmp_zern = calc_legendre_basis(nmodes, npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']

    zern_basis = zern_data['modes'][:nmodes]
    zern_basismat = zern_data['modesmat'][:nmodes]
    grid_mask = zern_data['mask']

    wf_zern_vec = 0
    grid_vec = grid_mask.reshape(-1)
    # if (fitweight != None):
    if (fitweight is not None):
        # Weighed LSQ fit with data. Only fit inside grid_mask

        # Multiply weight with binary mask, reshape to vector
        weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

        # LSQ fit with weighed data
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
        wf_zern_vec = np.linalg.lstsq((zern_basismat[:, grid_vec] * weight).T, wf_w.ravel())[0]
    else:
        # LSQ fit with data. Only fit inside grid_mask
        wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
        wf_zern_vec = np.linalg.lstsq(zern_basismat[:, grid_vec].T, wf_w.ravel(),rcond=None)[0]

    wf_zern_vec[:startmode-1] = 0

    # Calculates full Zernike phase & fitting error
    if (rec_zern):
        wf_zern_rec = calc_zernike_rec(wf_zern_vec, zern_data=zern_data, npix=wavefront.shape)
        fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
        fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
    else:
        wf_zern_rec = None
        fitdiff = None

    if (err != None):
        # For calculating scalar fitting qualities, only use the area inside the mask
        fitresid = fitdiff[grid_mask == True]
        err.append((fitresid**2.0).mean())
        err.append(np.abs(fitresid).mean())
        err.append(np.abs(fitresid).mean()**2.0)

    return (wf_zern_vec, wf_zern_rec, fitdiff)



def calc_legendre(leg_vec, npix, leg_data={}, mask=True):
    """
    Constructs wavefront with Legendre amplitudes **leg_vec**. Given vector **leg_vec** with the amplitude of Legendre
    modes, return the reconstructed wavefront with radius **rad**. This function uses **leg_data** as cache. If this is
    not given, it will be generated. See calc_legendre_basis() for details. If **mask** is True, set everything outside
    radius **rad** to zero, this is the default and will use orthogonal Legendre. If this is False, the modes will not
    be cropped.

    @param [in] leg_vec 1D vector of 2D Legendre amplitudes
    @param [in] npix array size to calculate the basis
    @param [in] leg_data 2D legendre basis cache
    @param [in] mask If True, set everything outside the Legendre aperture to zero, otherwise leave as is.
    @see See calc_legendre_basis() for details on **leg_data** cache and **mask**
    """

    from functools import reduce
    zern_data = leg_data
    zern_vec = leg_vec
    if (not ('modes' in zern_data.keys()) ):
        tmp_zern = calc_legendre_basis(len(zern_vec), npix)
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    # Compute Zernike basis if insufficient
    elif (len(zern_vec) > len(zern_data['modes'])):
        tmp_zern = calc_legendre_basis(len(zern_vec), npix)
        # This data already exists, overwrite it with new data
        zern_data['modes'] = tmp_zern['modes']
        zern_data['modesmat'] = tmp_zern['modesmat']
        zern_data['covmat'] = tmp_zern['covmat']
        zern_data['covmat_in'] = tmp_zern['covmat_in']
        zern_data['mask'] = tmp_zern['mask']
    zern_basis = zern_data['modes']

    gridmask = 1
    if (mask):
        gridmask = zern_data['mask']

    # Reconstruct the wavefront by summing modes
    return reduce(lambda x,y: x+y[1]*zern_basis[y[0]], enumerate(zern_vec), 0)


def legendre_2D(j, X, Y, norm=True):
    """
    Calculates the orthonormal rectangular Legendre mode j on a normalised rectangular grid **X** and **Y** given by:

    x_axis = np.linspace(-1, 1, npix_x)
    y_axis = np.linspace(-1, 1, npix_y)
    X, Y = np.meshgrid(x_axis, y_axis)

    @param [in] j indicates the 2D Polynomial order Q_j = L_l(X)*L_m(Y)
    @param [in] X - horizontal normalised coordinates
    @param [in] Y - vertical normalised coordinates
    @return 2D Legendre mode j with identical shape as X and Y

    """
    if j is 1:      # Piston
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(0, Y, norm))
    elif j is 2:    # x-tilt
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(0, Y, norm))
    elif j is 3:    # y-tilt
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(1, Y, norm))
    elif j is 4:    # x-defocus
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(0, Y, norm))
    elif j is 5:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(1, Y, norm))
    elif j is 6:    # y-defocus
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(2, Y, norm))
    elif j is 7:    # primary x-coma
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(0, Y, norm))
    elif j is 8:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(1, Y, norm))
    elif j is 9:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(2, Y, norm))
    elif j is 10:   # primary y-coma
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(3, Y, norm))
    elif j is 11:   # primary x-spherical
        return np.multiply(legendre_1D(4, X, norm), legendre_1D(0, Y, norm))
    elif j is 12:
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(1, Y, norm))
    elif j is 13:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(2, Y, norm))
    elif j is 14:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(3, Y, norm))
    elif j is 15:   # primary y-spherical
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(4, Y, norm))
    elif j is 16:   # secondary x-coma
        return np.multiply(legendre_1D(5, X, norm), legendre_1D(0, Y, norm))
    elif j is 17:
        return np.multiply(legendre_1D(4, X, norm), legendre_1D(1, Y, norm))
    elif j is 18:
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(2, Y, norm))
    elif j is 19:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(3, Y, norm))
    elif j is 20:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(4, Y, norm))
    elif j is 21:   # secondary y-coma
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(5, Y, norm))
    elif j is 22:   # secondary x-spherical
        return np.multiply(legendre_1D(6, X, norm), legendre_1D(0, Y, norm))
    elif j is 23:
        return np.multiply(legendre_1D(5, X, norm), legendre_1D(1, Y, norm))
    elif j is 24:
        return np.multiply(legendre_1D(4, X, norm), legendre_1D(2, Y, norm))
    elif j is 25:
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(3, Y, norm))
    elif j is 26:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(4, Y, norm))
    elif j is 27:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(5, Y, norm))
    elif j is 28:   # secondary y-spherical
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(6, Y, norm))
    elif j is 29:   # tertiary x-coma
        return np.multiply(legendre_1D(7, X, norm), legendre_1D(0, Y, norm))
    elif j is 30:
        return np.multiply(legendre_1D(6, X, norm), legendre_1D(1, Y, norm))
    elif j is 31:
        return np.multiply(legendre_1D(5, X, norm), legendre_1D(2, Y, norm))
    elif j is 32:
        return np.multiply(legendre_1D(4, X, norm), legendre_1D(3, Y, norm))
    elif j is 33:
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(4, Y, norm))
    elif j is 34:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(5, Y, norm))
    elif j is 35:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(6, Y, norm))
    elif j is 36:   # tertiary y-coma
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(7, Y, norm))
    elif j is 37:   # tertiary x-spherical
        return np.multiply(legendre_1D(8, X, norm), legendre_1D(0, Y, norm))
    elif j is 37:
        return np.multiply(legendre_1D(7, X, norm), legendre_1D(1, Y, norm))
    elif j is 38:
        return np.multiply(legendre_1D(6, X, norm), legendre_1D(2, Y, norm))
    elif j is 39:
        return np.multiply(legendre_1D(5, X, norm), legendre_1D(3, Y, norm))
    elif j is 40:
        return np.multiply(legendre_1D(4, X, norm), legendre_1D(4, Y, norm))
    elif j is 41:
        return np.multiply(legendre_1D(3, X, norm), legendre_1D(5, Y, norm))
    elif j is 42:
        return np.multiply(legendre_1D(2, X, norm), legendre_1D(6, Y, norm))
    elif j is 43:
        return np.multiply(legendre_1D(1, X, norm), legendre_1D(7, Y, norm))
    elif j is 44:   # tertiary y-spherical
        return np.multiply(legendre_1D(0, X, norm), legendre_1D(8, Y, norm))


def legendre_1D(Ln, X, norm=True):
    """
    Calculates the 1D Legendre polynomials on a X grid ranging from -1 to 1. The polynomials can be obtained using the
    Rodrigues formula (https://en.wikipedia.org/wiki/Rodrigues%27_formula): Ln = 1/(2^n n!) * d^n(x^2-1)^n/dx^n, where
    d^n/dx^n indicates the nth derivative of (x^2-1)^n

    @param [in] Ln: polynomial index
    @param [in] X: grid ranging from [-1,1]. np.linspace(-1,1,npix) or np.meshgrid(-1,1,npix)
    @param [in] norm: puts the normal in orthonormal, otherwise the base is just orthogonal
    @return 1D legendre polynomial calculated over a grid.
    """

    k = 1
    if norm is True:
        k = np.sqrt(2*Ln + 1)

    if Ln is 0:     # Piston
        return np.ones(X.shape) * k
    elif Ln is 1:     # Tilt
        return X
    elif Ln is 2:     # Defocus
        return (3*X**2 -1)/2
    elif Ln is 3:     # Coma
        return (5*X**3 - 3*X)/2
    elif Ln is 4:     # Spherical aberration
        return (35*X**4 - 30*X**2 + 3)/8
    elif Ln is 5:     # Secondary coma
        return (63*X**5 - 70*X**3 + 15*X)/8
    elif Ln is 6:     # Secondary spherical aberration
        return (231*X**6 - 315*X**4 + 105*X**2 - 5)/16
    elif Ln is 7:     # Tertiary coma
        return (429*X**7 - 693*X**5 + 315*X**3 - 35*X)/16
    elif Ln is 8:     # Tertiary spherical aberration
        return (6435*X**8 - 12012*X**6 + 6930*X**4 -1260*X**2 + 35)/128
