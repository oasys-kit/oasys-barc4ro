#!/usr/bin/python
# coding: utf-8

###################################################################################
# barc4RefractiveOptics
# Authors/Contributors: Rafael Celestre, Oleg Chubar, Manuel Sanchez del Rio
# Rafael.Celestre@esrf.eu
# creation: 24.06.2019
# last update: 06.01.2023 (v0.5)
#
# Check documentation:
# Celestre, R. et al. (2020). Recent developments in x-ray lenses modelling with SRW.
# Proc. SPIE 11493, Advances in Computational Methods for X-Ray Optics V, 11493-17.
# arXiv:2007.15461 [physics.optics] - https://arxiv.org/abs/2007.15461
#
# Copyright (c) 2019-2023 European Synchrotron Radiation Facility
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

try:
    from oasys_srw.srwlib import SRWLRadMesh, SRWLOptT
except:
    try:
        from srwlib import SRWLRadMesh, SRWLOptT
    except:
        pass

from barc4ro.projected_thickness import *

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Optical Elements
# ----------------------------------------------------------------------------------------------------------------------

def srwl_opt_setup_CRL(_foc_plane, _delta, _atten_len, _shape, _apert_h, _apert_v, _r_min, _n, _wall_thick=0, _xc=0,
                       _yc=0, _e_start=0, _e_fin=0, _nx=1001, _ny=1001, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0,
                       _offst_ffs_x=0, _offst_ffs_y=0, _tilt_ffs_x=0, _tilt_ffs_y=0, _ang_rot_ez_ffs=0, _wt_offst_ffs=0,
                       _offst_bfs_x=0, _offst_bfs_y=0, _tilt_bfs_x=0, _tilt_bfs_y=0, _ang_rot_ez_bfs=0, _wt_offst_bfs=0,
                       _axis_x=None, _axis_y=None, isdgr=False, _aperture=None):
    """
    Setup Transmission type Optical Element which simulates Compound Refractive Lens (CRL)
    :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _shape: 1- parabolic, 2- circular (spherical), 3- elliptical (not implemented), 4- Cartesian oval (not implemented)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _r_min: radius (on tip of parabola for parabolic shape) [m]
    :param _n: number of lenses (/"holes")
    :param _wall_thick:  min. wall thickness between "holes" [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _e_start: initial photon energy
    :param _e_fin: final photon energy
    :param _nx: number of points vs horizontal position to represent the transmission element
    :param _ny: number of points vs vertical position to represent the transmission element
    :param _ang_rot_ex: angle [rad] of full CRL rotation about horizontal axis
    :param _ang_rot_ey: angle [rad] of full CRL rotation about vertical axis
    :param _ang_rot_ez: angle [rad] of full CRL rotation about longitudinal axis
    :param _offst_ffs_x: lateral offset in the horizontal axis of the front focusing surface [m]
    :param _offst_ffs_y: lateral offset in the vertical axis of the front focusing surface [m]
    :param _tilt_ffs_x: angle [rad] of the parabolic front surface rotation about horizontal axis
    :param _tilt_ffs_y: angle [rad] of the parabolic front surface rotation about horizontal axis
    :param _ang_rot_ez_ffs: angle [rad] of the parabolic front surface rotation about the longitudinal axis
    :param _wt_offst_ffs: excess penetration [m] of the front parabola to be added to _wall_thick
    :param _offst_bfs_x: lateral offset in the horizontal axis of the back focusing surface [m]
    :param _offst_bfs_y: lateral offset in the back axis of the front focusing surface [m]
    :param _tilt_bfs_x: angle [rad] of the parabolic front back rotation about horizontal axis
    :param _tilt_bfs_y: angle [rad] of the parabolic front back rotation about horizontal axis
    :param _ang_rot_ez_bfs: angle [rad] of the parabolic back surface rotation about the longitudinal axis
    :param _wt_offst_bfs: excess penetration [m] of the back parabola to be added to _wall_thick (negative or positive values)
    :param isdgr: boolean for determining if angles are in degree or in radians (default)
    :param _aperture: specifies the type of aperture: circular ('c') or square ('s')
    :return: transmission (SRWLOptT) type optical element which simulates a CRL
    """

    foc_len = _r_min/(_n*_delta*2)
    print('Optical Element Setup: CRL Focal Length:', foc_len, 'm')
    fx = 1e+23
    fy = 1e+23

    if _foc_plane != 1:
        fy = foc_len
    if _foc_plane != 2:
        fx = foc_len

    if _foc_plane == 3 and _aperture is None:
        _aperture = 'c'
    if (_foc_plane == 1 or _foc_plane == 2) and _aperture is None:
        _aperture = 'r'

    if _n == 0.5:
        surfs = 1
        _n = 1
    else:
        surfs = 2

    x, y, thcknss = proj_thick_crl_2D(_foc_plane=_foc_plane, _shape=_shape, _apert_h=_apert_h, _apert_v=_apert_v,
                                      _r_min=_r_min, _n=surfs, _wall_thick=_wall_thick, _xc=_xc, _yc=_yc, _nx=_nx,
                                      _ny=_ny, _ang_rot_ex=_ang_rot_ex, _ang_rot_ey=_ang_rot_ey, _ang_rot_ez=_ang_rot_ez,
                                      _offst_ffs_x=_offst_ffs_x, _offst_ffs_y=_offst_ffs_y, _tilt_ffs_x=_tilt_ffs_x,
                                      _tilt_ffs_y=_tilt_ffs_y, _ang_rot_ez_ffs=_ang_rot_ez_ffs, _wt_offst_ffs=_wt_offst_ffs,
                                      _offst_bfs_x=_offst_bfs_x, _offst_bfs_y=_offst_bfs_y, _tilt_bfs_x=_tilt_bfs_x,
                                      _tilt_bfs_y=_tilt_bfs_y, _ang_rot_ez_bfs=_ang_rot_ez_bfs,
                                      _wt_offst_bfs=_wt_offst_bfs, isdgr=isdgr, project=True, _axis_x=_axis_x,
                                      _axis_y=_axis_y, _aperture=_aperture)

    _xc = 0.0
    _yc = 0.0

    _ny, _nx = thcknss.shape

    thcknss *= _n

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    # for debugging:
    # return x, y, thcknss, amplitude_transmission, optical_path_diff

    arTr = np.empty((2 * _nx * _ny), dtype=thcknss.dtype)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, x[-1]-x[0], y[-1]-y[0], _arTr=arTr, _extTr=1, _Fx=fx, _Fy=fy, _x=_xc, _y=_yc)


def srwl_opt_setup_axicon(_foc_plane, _delta, _atten_len, _shape, _apert_h, _apert_v, _h, _n, _wall_thick=0, _xc=0,
                          _yc=0, _e_start=0, _e_fin=0, _nx=1001, _ny=1001, _axis_x=None, _axis_y=None):
    """
    Setup Transmission type Optical Element which simulates an axicon
    :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _shape: 'p' - positive or 'n' - negative
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _h: structure height [m]
    :param _n: number of axicons
    :param _wall_thick:  substrat thickness [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _e_start: initial photon energy
    :param _e_fin: final photon energy
    :param _nx: number of points vs horizontal position to represent the transmission element
    :param _ny: number of points vs vertical position to represent the transmission element
    :return: transmission (SRWLOptT) type optical element which simulates an axicon
    """

    fx = 1e+23
    fy = 1e+23

    x, y, thcknss = proj_thick_axicon_2D(_foc_plane, _shape, _apert_h, _apert_v, _h,
                                         _wall_thick, _xc, _yc, _nx, _ny, _axis_x, _axis_y)

    _xc = 0.0
    _yc = 0.0

    _ny, _nx = thcknss.shape

    thcknss *= _n

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    arTr = np.empty((2 * _nx * _ny), dtype=thcknss.dtype)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, x[-1]-x[0], y[-1]-y[0], _arTr=arTr, _extTr=1, _Fx=fx, _Fy=fy, _x=_xc, _y=_yc)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Arbitrary surfaces
# ----------------------------------------------------------------------------------------------------------------------

def srwl_opt_setup_arbitrary_surf(_height_prof_data, _mesh, _delta, _atten_len, _wall_thick=0.0, _amp_coef=1, _xc=0,
                                 _yc=0, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0, _fx=1e23, _fy=1e23, isdgr=False):
    """
    Setup Transmission type Optical Element which simulates Compound Refractive Lens (CRL) figure errors in [m] from a
    .dat file format is defined in srwl_uti_save_intens_ascii.
    :param _height_prof_data: Figure error array data in [m] from an ASCII file (format is defined in srwl_uti_save_intens_ascii)
    :param _mesh: mesh vs photon energy, horizontal and vertical positions (SRWLRadMesh type) on which errors were measured
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _wall_thick: constant offset [m]
    :param _amp_coef: multiplicative factor
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _ang_rot_ex: angle [rad] of CRL rotation about horizontal axis
    :param _ang_rot_ey: angle [rad] of CRL rotation about vertical axis
    :param _ang_rot_ez: angle [rad] of CRL rotation about longitudinal axis
    :param _fx: approximate horizontal focal length of the free form, if any
    :param _fy: approximate vertical focal length of the free form, if any
    :param isdgr: boolean for determining if angles are in degree or in radians (default)
    :return: transmission (SRWLOptT) type optical element which simulates CRL figure errors
    """

    _height_prof_data = np.reshape(_height_prof_data, (_mesh.ny, _mesh.nx))
    _height_prof_data *= _amp_coef

    dx = (_mesh.xFin - _mesh.xStart) / _mesh.nx
    dy = (_mesh.yFin - _mesh.yStart) / _mesh.ny

    pad_y = int(_mesh.ny*0.1)
    pad_x = int(_mesh.nx*0.1)

    thcknss = np.pad(_height_prof_data, ((pad_y, pad_y),(pad_x, pad_x)), 'constant', constant_values=0)

    if _ang_rot_ex != 0 or _ang_rot_ey != 0:

        _ny, _nx = thcknss.shape
        xStart = - (dx * (_nx - 1)) / 2.0
        xFin = xStart + dx * (_nx - 1)
        yStart = - (dy * (_ny - 1)) / 2.0
        yFin = yStart + dy * (_ny - 1)
        _ny, _nx = thcknss.shape
        x = np.linspace(xStart, xFin, _nx)
        y = np.linspace(yStart, yFin, _ny)
        tilt = np.zeros(thcknss.shape)
        rz = thcknss
        rx, ry, rz = at_rotate_2D_nested_loop(x, y, rz, th_x=_ang_rot_ex, th_y=_ang_rot_ey, isdgr=isdgr)
        thcknss = rz - tilt

    _ny, _nx = thcknss.shape
    xStart = - (dx * (_nx - 1)) / 2.0
    xFin = xStart + dx * (_nx - 1)
    yStart = - (dy * (_ny - 1)) / 2.0
    yFin = yStart + dy * (_ny - 1)

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    arTr = np.empty((2 * _nx * _ny), dtype=float)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, xFin-xStart, yFin-yStart, _arTr=arTr, _extTr=1, _Fx=_fx, _Fy=_fy, _x=_xc, _y=_yc)


def srwl_opt_setup_polynomial_surf(_z_coeffs, _pol, _delta, _atten_len, _apert_h, _apert_v, _xc=0, _yc=0, _nx=1001, _ny=1001):

    """
    If _z_coeffs is a single number, it refers to the piston value. So an array or random numbers representing the
    coefficients of the polynomials will be generated from -1 to 1 and later normalised to _z_coeffs. If _zcoeffs is a
    list, the function will return a surface based on it.
    :param _z_coeffs: either a list of polynomial coefficients or the total RMS value of the surface errors [m]
    :param _pol: 'c' - circular Zernike; 'r' - rectangular polynomials; 'l' - legendre polynomials
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc:  vertical coordinate of center [m]
    :param _nx: number of points vs horizontal position to represent the transmission element
    :param _ny: number of points vs vertical position to represent the transmission element
    :return: transmission (SRWLOptT) type optical element which simulates CRL figure errors
    """

    fx = 1e+23
    fy = 1e+23

    x, y, thcknss = polynomial_surface_2D(_z_coeffs=_z_coeffs, _pol=_pol, _apert_h=_apert_h, _apert_v=_apert_v, _nx=_nx, _ny=_ny)

    dx = _apert_h/_nx
    dy = _apert_v/_ny

    _xc = 0.0
    _yc = 0.0

    pad_y = int(_ny*0.1)
    pad_x = int(_nx*0.1)

    thcknss = np.pad(thcknss, ((pad_y, pad_y),(pad_x, pad_x)), 'constant', constant_values=0)

    _ny, _nx = thcknss.shape
    xStart = - (dx * (_nx - 1)) / 2.0
    xFin = xStart + dx * (_nx - 1)
    yStart = - (dy * (_ny - 1)) / 2.0
    yFin = yStart + dy * (_ny - 1)

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    # # for debugging:
    # x = np.linspace(xStart, xFin, thcknss.shape[1])
    # y = np.linspace(yStart, yFin, thcknss.shape[0])
    # return x, y, thcknss , amplitude_transmission, optical_path_diff

    arTr = np.empty((2 * _nx * _ny), dtype=thcknss.dtype)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, xFin-xStart, yFin-yStart, _arTr=arTr, _extTr=1, _Fx=fx, _Fy=fy, _x=_xc, _y=_yc)


def srwl_opt_setup_fractal_surf(_sigma, _psd_slope, _pix_size, _delta, _atten_len, _apert_h, _apert_v, _xc=0, _yc=0,
                                _qr=0, _seed=None, _C=None, _dist=0):
    """
    Setup Transmission type Optical Element which simulates a rough surface at normal incidence in [m] with a
    pre-determined PSD. The PSD can be defined by either the rms value of the roughness(_sigma), _psd_slope and
    roll-off freq. (_qr); or by a directly calculated 2D PSD (_C).

    :param _sigma: root-mean-square roughness Rq(m)
    :param _psd_slope: PSD exponent = -2(H+1); Hurst exponent 0<= H <= 1, fractal dimension D = 3-H
    :param _pix_size: pixel size in [m] for the resulting surface
    :param _delta: refractive index decrement
    :param _atten_len: attenuation length [m]
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/PixelWidth) - Nyquist freq.
    :param _seed: seed for random initialisation
    :param _C: pre-calculated 2D psd where qx and qy respect the limits imposed by _pix_size, _apert_h and _apert_v
    :param _dist: -1 for phase = 0, 0 for uniform phase rand. distribution, 1 for rand. Gaussian dist.
    :return: transmission (SRWLOptT) type optical element which simulates a rough surface
    """

    nx = int(_apert_h / _pix_size)
    ny = int(_apert_v / _pix_size)

    height_prof_data, y, x = fractal_surf(_sigma, _psd_slope, _pix_size, nx, ny, _qr=_qr, _dist=_dist,
                                                   _seed=_seed, _psd=False, _C=_C)

    pad_y = int(ny * 0.15)
    pad_x = int(nx * 0.15)

    thcknss = np.pad(height_prof_data, ((pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=0)

    ny, nx = thcknss.shape
    xStart = - (_pix_size * (nx - 1)) / 2.0
    xFin = xStart + _pix_size * (nx - 1)
    yStart = - (_pix_size * (ny - 1)) / 2.0
    yFin = yStart + _pix_size * (ny - 1)

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    arTr = np.empty((2 * nx * ny), dtype=float)
    arTr[0::2] = np.reshape(amplitude_transmission, (nx * ny))
    arTr[1::2] = np.reshape(optical_path_diff, (nx * ny))

    return SRWLOptT(nx, ny, xFin - xStart, yFin - yStart, _arTr=arTr, _extTr=1, _x=_xc, _y=_yc)



