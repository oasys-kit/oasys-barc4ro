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
from numbers import Number
import random
from scipy.interpolate import interp1d, interp2d
import warnings

# try:
#     # you can add the SRW path here:
#     import sys
#     sys.path.insert(0, '../../srw_python')
#     from srwlib import SRWLRadMesh, SRWLOptT
# except:
#     print("SRW not in the PYTHONPATH try adding to ~/.bashrc:")
#     print('~$ export PYTHONPATH="${PYTHONPATH}:/my/path/to/SRW"')
#     print("or copy this file to your srw_python folder")

try:
    from srwlib import SRWLRadMesh, SRWLOptT
except:
    from oasys_srw.srwlib import SRWLRadMesh, SRWLOptT

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Optical Elements
# ----------------------------------------------------------------------------------------------------------------------

def srwl_opt_setup_CRL(_foc_plane, _delta, _atten_len, _shape, _apert_h, _apert_v, _r_min, _n, _wall_thick=0, _xc=0,
                       _yc=0, _e_start=0, _e_fin=0, _nx=1001, _ny=1001, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0,
                       _offst_ffs_x=0, _offst_ffs_y=0, _tilt_ffs_x=0, _tilt_ffs_y=0, _ang_rot_ez_ffs=0, _wt_offst_ffs=0,
                       _offst_bfs_x=0, _offst_bfs_y=0, _tilt_bfs_x=0, _tilt_bfs_y=0, _ang_rot_ez_bfs=0, _wt_offst_bfs=0,
                       isdgr=False):
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
    :return: ransmission (SRWLOptT) type optical element which simulates a CRL
    """

    foc_len = _r_min/(_n*_delta*2)
    print('Optical Element Setup: CRL Focal Length:', foc_len, 'm')
    fx = 1e+23
    fy = 1e+23

    if(_foc_plane != 1):
        fy = foc_len
    if(_foc_plane != 2):
        fx = foc_len

    if _n == 0.5:
        surfs = 1
        _n = 1
    else:
        surfs = 2

    x, y, thcknss = proj_thick_2D_crl(_foc_plane=_foc_plane, _shape=_shape, _apert_h=_apert_h, _apert_v=_apert_v,
                                      _r_min=_r_min, _n=surfs, _wall_thick=_wall_thick, _xc=_xc, _yc=_yc, _nx=_nx,
                                      _ny=_ny, _ang_rot_ex=_ang_rot_ex, _ang_rot_ey=_ang_rot_ey, _ang_rot_ez=_ang_rot_ez,
                                      _offst_ffs_x=_offst_ffs_x, _offst_ffs_y=_offst_ffs_y, _tilt_ffs_x=_tilt_ffs_x,
                                      _tilt_ffs_y=_tilt_ffs_y, _ang_rot_ez_ffs=0, _wt_offst_ffs=_wt_offst_ffs,
                                      _offst_bfs_x=_offst_bfs_x, _offst_bfs_y=_offst_bfs_y, _tilt_bfs_x=_tilt_bfs_x,
                                      _tilt_bfs_y=_tilt_bfs_y, _ang_rot_ez_bfs=_ang_rot_ez_bfs,
                                      _wt_offst_bfs=_wt_offst_bfs, isdgr=isdgr, project=True, _axis_x=None,
                                      _axis_y=None, _aperture='c')


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


def srwl_opt_setup_CRL_metrology(_height_prof_data, _mesh, _delta, _atten_len, _wall_thick=0.0, _amp_coef=1, _xc=0,
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
    _height_prof_data *=_amp_coef

    dx = (_mesh.xFin - _mesh.xStart) / _mesh.nx
    dy = (_mesh.yFin - _mesh.yStart) / _mesh.ny


    pad_y = int(_mesh.ny*0.1)
    pad_x = int(_mesh.nx*0.1)

    thcknss = np.pad(_height_prof_data, ((pad_y, pad_y),(pad_x, pad_x)), 'constant', constant_values=0)

    if _ang_rot_ex != 0 or _ang_rot_ey != 0:
        print('OI')
        _ny, _nx = thcknss.shape
        xStart = - (dx * (_nx - 1)) / 2.0
        xFin = xStart + dx * (_nx - 1)
        yStart = - (dy * (_ny - 1)) / 2.0
        yFin = yStart + dy * (_ny - 1)
        _ny, _nx = thcknss.shape
        x = np.linspace(_mesh.xStart, _mesh.xFin, _nx)
        y = np.linspace(_mesh.yStart, _mesh.yFin, _ny)
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

    arTr = np.empty((2 * _nx * _ny), dtype=np.float)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, xFin-xStart, yFin-yStart, _arTr=arTr, _extTr=1, _Fx=_fx, _Fy=_fy, _x=_xc, _y=_yc)


def srwl_opt_setup_CRL_errors(_z_coeffs, _pol, _delta, _atten_len, _apert_h, _apert_v, _xc=0, _yc=0, _nx=1001, _ny=1001):
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

    return SRWLOptT(_nx, _ny, xFin-xStart, yFin-yStart, _arTr=arTr, _extTr=1, _Fx=1e23, _Fy=1e23, _x=_xc, _y=_yc)

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Projected thickness calculation
# ----------------------------------------------------------------------------------------------------------------------

def proj_thick_1D_crl(_shape, _apert_h, _r_min, _n=2, _wall_thick=0, _xc=0, _nx=1001, _ang_rot_ex=0, _offst_ffs_x=0,
                      _tilt_ffs_x=0, _wt_offst_ffs=0, _offst_bfs_x=0, _tilt_bfs_x=0, _wt_offst_bfs=0, isdgr=False,
                      project=True, _axis=None):
    """
    1D X-ray lens (CRL) thickness profile
    :param _shape: 1- parabolic, 2- circular (future), 3- elliptical (future), 4- Cartesian oval (future)
    :param _apert_h: aperture size [m]
    :param _r_min: radius (on tip of parabola for parabolic shape) [m]
    :param _n: number of refracting surfaces. Either '1' or '2'.
    :param _wall_thick: min. wall thickness between "holes" [m]
    :param _xc: coordinate of center [m]
    :param _nx: number of points to represent the transmission element
    :param _ang_rot_ex: angle [rad] of full CRL rotation about horizontal axis
    :param _offst_ffs_x: lateral offeset in the horizontal axis of the front focusing surface (ffs) [m]
    :param _tilt_ffs_x: angle [rad] of the parabolic ffs rotation about horizontal axis
    :param _wt_offst_ffs: excess penetration [m] of the front parabola to be added to _wall_thick (negative or positive values)
    :param _offst_bfs_x: lateral offeset in the horizontal axis of the back focusing surface (bfs) [m]
    :param _tilt_bfs_x: angle [rad] of the parabolic bfs rotation about horizontal axis
    :param _wt_offst_bfs: excess penetration [m] of the back parabola to be added to _wall_thick (negative or positive values)
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :param project: boolean. project=True necessary for using the profile as a transmission element
    :param _axis: forces the lens to be calculated on a given grid - avoids having to interpolate different calculations to the same grid
    :return: thickness profile
    """
    if _shape == 1:
        # ------------- calculates the max inclination before obscuration happens
        if isdgr:
            alpha_max = (np.pi/2 - np.arctan(_apert_h/2/_r_min)) * 180/np.pi
        else:
             alpha_max = (np.pi/2 - np.arctan(_apert_h/2/_r_min))

        if _ang_rot_ex>0.98*alpha_max or _tilt_ffs_x>0.98*alpha_max or _tilt_bfs_x>0.98*alpha_max:
            print('>>> Tilt angle causes obscuration - error in interpolation on the edges od the lens may be present')

        k = 0.6
        L_half = (_apert_h / 2) ** 2 / 2 / _r_min + _wall_thick / 2

        if _axis is None:
            if _xc != 0 or _ang_rot_ex != 0 or _tilt_ffs_x != 0 or _offst_ffs_x != 0 or _tilt_bfs_x != 0 or _offst_bfs_x != 0:
                theta = np.amax([np.abs(_ang_rot_ex), np.abs(_tilt_ffs_x), np.abs(_tilt_bfs_x)])
                xn, zn = at_rotate_1D(-_apert_h/2, L_half, th=theta, isdgr=isdgr, project=False)
                xp, zp = at_rotate_1D( _apert_h/2, L_half, th=theta, isdgr=isdgr, project=False)
                k *= np.amax([np.abs(xn)/(_apert_h/2),np.abs(xp)/(_apert_h/2),1+np.abs(_xc)/(_apert_h/2),
                              1+np.abs(_offst_ffs_x)/(_apert_h/2),1+np.abs(_offst_bfs_x)/(_apert_h/2)])
            x = np.linspace(-k * _apert_h, k * _apert_h, _nx)
        else:
            x = _axis
        # ========================================================
        # ========== Front Focusing Surface calculations =========
        # ========================================================

        # ------------- new aperture resulting from different penetration depths
        neg_values_ffs = False
        if _wt_offst_ffs != 0:
            _wall_thick_ffs = _wall_thick + _wt_offst_ffs*2
            _apert_h_ffs = 2*np.sqrt(2*(L_half-_wall_thick_ffs/2)*_r_min)
            if _wall_thick_ffs < 0:
                neg_values_ffs = True
        else:
            _wall_thick_ffs = _wall_thick
            _apert_h_ffs = _apert_h

        # ------------- calculation of thickness profile in projection approximation
        delta_z_ffs = (x - _xc - _offst_ffs_x) ** 2 / _r_min / 2 + _wall_thick_ffs/2
        delta_z_ffs[x - _xc - _offst_ffs_x< -0.5 * _apert_h_ffs] = L_half
        delta_z_ffs[x - _xc - _offst_ffs_x>  0.5 * _apert_h_ffs] = L_half

        if neg_values_ffs:
            f_mask = np.zeros(x.shape, dtype=bool)
            f_mask[delta_z_ffs < 0] = True
            delta_z_ffs[f_mask] = 0

        # ------------- rotation of the front focusing parabolic surface
        if _tilt_ffs_x != 0:
            xn, zn = at_rotate_1D(-_apert_h_ffs/2 + _xc + _offst_ffs_x, L_half, th=_tilt_ffs_x, isdgr=isdgr, project=False)
            xp, zp = at_rotate_1D( _apert_h_ffs/2 + _xc + _offst_ffs_x, L_half, th=_tilt_ffs_x, isdgr=isdgr, project=False)
            rx, rz = at_rotate_1D(x, delta_z_ffs, th=_tilt_ffs_x, isdgr=isdgr, project=True)
            p = np.polyfit((xn, xp), (zn, zp), 1)
            rz -= p[0] * x
            rz[x > xp] = L_half
            rz[x < xn] = L_half
            rz[rz > L_half] = L_half
            delta_z_ffs = rz

        # ------------- full rotation of the CRL_half applied to the full front focusing surface
        if _ang_rot_ex != 0:

            rx, rz = at_rotate_1D(x, delta_z_ffs, th=_ang_rot_ex, isdgr=isdgr, project=True)

            if _tilt_ffs_x != 0:
                xn, zn = at_rotate_1D(xn, L_half, th=_ang_rot_ex, isdgr=isdgr, project=False)
                xp, zp = at_rotate_1D(xp, L_half, th=_ang_rot_ex, isdgr=isdgr, project=False)
            else:
                xn, zn = at_rotate_1D(-_apert_h/2 + _xc + _offst_ffs_x, L_half, th=_ang_rot_ex, isdgr=isdgr, project=False)
                xp, zp = at_rotate_1D( _apert_h/2 + _xc + _offst_ffs_x, L_half, th=_ang_rot_ex, isdgr=isdgr, project=False)

            # projected thickness increase of the non-focusing surface
            p = np.polyfit((xn, xp), (zn, zp), 1)
            f_top = p[0] * x + p[1]
            f_mask_p = np.ones(x.shape, dtype=bool)
            f_mask_p[x >= xp] = False
            f_mask_p[x <= xn] = False
            f_mask_l = np.logical_not(f_mask_p)
            f_top[f_mask_p] = 0
            rz[f_mask_l] = 0
            rz += f_top

            # for projection approximation
            if project:
                rz -= p[0] * x
            delta_z_ffs = rz

        # ========================================================
        # ========== Back Focusing Surface calculations ==========
        # ========================================================

        if _n == 2:
            # ------------- new aperture resulting from different penetration depths
            neg_values_bfs = False
            if _wt_offst_bfs != 0:
                _wall_thick_bfs = _wall_thick + _wt_offst_bfs * 2
                _apert_h_bfs = 2 * np.sqrt(2 * (L_half - _wall_thick_bfs / 2) * _r_min)
                if _wall_thick_bfs < 0:
                    neg_values_bfs = True
            else:
                _wall_thick_bfs = _wall_thick
                _apert_h_bfs = _apert_h

            # ------------- calculation of thickness profile in projection approximation

            delta_z_bfs = (x - _xc - _offst_bfs_x) ** 2 / _r_min / 2 + _wall_thick_bfs / 2
            delta_z_bfs[x - _xc - _offst_bfs_x < -0.5 * _apert_h_bfs] = L_half
            delta_z_bfs[x - _xc - _offst_bfs_x > 0.5 * _apert_h_bfs] = L_half

            if neg_values_bfs:
                f_mask = np.zeros(x.shape, dtype=bool)
                f_mask[delta_z_bfs < 0] = True
                delta_z_bfs[f_mask] = 0

            # ------------- rotation of the back focusing parabolic surface
            if _tilt_bfs_x != 0:
                xn, zn = at_rotate_1D(-_apert_h_bfs / 2 + _xc + _offst_bfs_x, L_half, th=-_tilt_bfs_x, isdgr=isdgr, project=False)
                xp, zp = at_rotate_1D( _apert_h_bfs / 2 + _xc + _offst_bfs_x, L_half, th=-_tilt_bfs_x, isdgr=isdgr, project=False)
                rx, rz = at_rotate_1D(x, delta_z_bfs, th=-_tilt_bfs_x, isdgr=isdgr, project=True)
                p = np.polyfit((xn, xp), (zn, zp), 1)
                rz -= p[0] * x
                rz[x > xp] = L_half
                rz[x < xn] = L_half
                rz[rz > L_half] = L_half
                delta_z_bfs = rz

            # ------------- full rotation of the CRL applied to the full back focusing surface
            if _ang_rot_ex != 0:

                rx, rz = at_rotate_1D(x, delta_z_bfs, th=-_ang_rot_ex, isdgr=isdgr, project=True)

                if _tilt_bfs_x != 0:
                    xn, zn = at_rotate_1D(xn, L_half, th=-_ang_rot_ex, isdgr=isdgr, project=False)
                    xp, zp = at_rotate_1D(xp, L_half, th=-_ang_rot_ex, isdgr=isdgr, project=False)
                else:
                    xn, zn = at_rotate_1D(-_apert_h / 2 + _xc + _offst_bfs_x, L_half, th=-_ang_rot_ex, isdgr=isdgr, project=False)
                    xp, zp = at_rotate_1D( _apert_h / 2 + _xc + _offst_bfs_x, L_half, th=-_ang_rot_ex, isdgr=isdgr, project=False)

                # projected thickness increase of the non-focusing surface
                p = np.polyfit((xn, xp), (zn, zp), 1)
                f_top = p[0] * x + p[1]
                f_mask_p = np.ones(x.shape, dtype=bool)
                f_mask_p[x >= xp] = False
                f_mask_p[x <= xn] = False
                f_mask_l = np.logical_not(f_mask_p)
                f_top[f_mask_p] = 0
                rz[f_mask_l] = 0
                rz += f_top

                # for projection approximation
                if project:
                    rz -= p[0] * x
                delta_z_bfs = rz
            return x, delta_z_ffs+delta_z_bfs
        return x, delta_z_ffs


def proj_thick_2D_crl(_foc_plane, _shape, _apert_h, _apert_v, _r_min, _n, _wall_thick=0, _xc=0, _yc=0, _nx=1001,
                      _ny=1001,_ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0, _offst_ffs_x=0, _offst_ffs_y=0,
                      _tilt_ffs_x=0, _tilt_ffs_y=0, _ang_rot_ez_ffs=0, _wt_offst_ffs=0, _offst_bfs_x=0, _offst_bfs_y=0,
                      _tilt_bfs_x=0, _tilt_bfs_y=0, _ang_rot_ez_bfs=0,_wt_offst_bfs=0, isdgr=False, project=True,
                      _axis_x=None, _axis_y=None, _aperture=None):
    """

    :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
    :param _shape: 1- parabolic, 2- circular (spherical), 3- elliptical (not implemented), 4- Cartesian oval (not implemented)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _r_min: radius (on tip of parabola for parabolic shape) [m]
    :param _n: number of lenses (/"holes")
    :param _wall_thick:  min. wall thickness between "holes" [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
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
    :param project: boolean. project=True necessary for using the profile as a transmission element
    :param _axis_x: forces the lens to be calculated on a given grid - avoids having to interpolate different calculations to the same grid
    :param _axis_y: forces the lens to be calculated on a given grid - avoids having to interpolate different calculations to the same grid
    :param _aperture: specifies the type of aperture: circular or square
    :return: thickness profile
    """

    if _foc_plane == 1:
        _apert = _apert_h
    elif _foc_plane == 2:
        _apert = _apert_v
    else:
        _apert_v = _apert_h
        _apert = _apert_h

    if _shape == 1:
        # ------------- calculates the max inclination before obscuration happens
        if isdgr:
            alpha_max = (np.pi/2 - np.arctan(_apert_h/2/_r_min)) * 180/np.pi
        else:
             alpha_max = (np.pi/2 - np.arctan(_apert_h/2/_r_min))

        if _ang_rot_ex>0.98*alpha_max or _tilt_ffs_x>0.98*alpha_max or _tilt_bfs_x>0.98*alpha_max:
            print('>>> Tilt angle causes obscuration in the horizontal direction  - error in interpolation '
                  'on the edges of the lens may be present')

        if _ang_rot_ey>0.98*alpha_max or _tilt_ffs_y>0.98*alpha_max or _tilt_bfs_y>0.98*alpha_max:
            print('>>> Tilt angle causes obscuration in the vertical direction  - error in interpolation '
                  'on the edges of the lens may be present')

        L_half = (_apert / 2) ** 2 / 2 / _r_min + _wall_thick / 2

        # horizontal axis calculation
        k = 0.6
        axis_x = False
        if _axis_x is None:
            if _xc != 0 or _ang_rot_ex != 0 or _tilt_ffs_x != 0 or _offst_ffs_x != 0 or _tilt_bfs_x != 0 or _offst_bfs_x!= 0:
                theta = np.amax([np.abs(_ang_rot_ex), np.abs(_tilt_ffs_x), np.abs(_tilt_bfs_x)])
                xxn, xzn = at_rotate_1D(-_apert_h/2, L_half, th=theta, isdgr=isdgr, project=False)
                xxp, xzp = at_rotate_1D( _apert_h/2, L_half, th=theta, isdgr=isdgr, project=False)
                k *= np.amax([np.abs(xxn)/(_apert_h/2),np.abs(xxp)/(_apert_h/2),1+np.abs(_xc)/(_apert_h/2),
                              1+np.abs(_offst_ffs_x)/(_apert_h/2),1+np.abs(_offst_bfs_x)/(_apert_h/2)])
            kx = k
            #x = np.linspace(-k * _apert_h, k * _apert_h, _nx)
        else:
            x = _axis_x
            axis_x = True
        k = 0.6
        axis_y = False
        if _axis_y is None:
            if _yc != 0 or _ang_rot_ey != 0 or _tilt_ffs_y != 0 or _offst_ffs_y != 0 or _tilt_bfs_y != 0 or _offst_bfs_y!= 0:
                theta = np.amax([np.abs(_ang_rot_ey), np.abs(_tilt_ffs_y), np.abs(_tilt_bfs_y)])
                yxn, yzn = at_rotate_1D(-_apert_v/2, L_half, th=theta, isdgr=isdgr, project=False)
                yxp, yzp = at_rotate_1D( _apert_v/2, L_half, th=theta, isdgr=isdgr, project=False)
                k *= np.amax([np.abs(yxn)/(_apert_v/2),np.abs(yxp)/(_apert_v/2),1+np.abs(_yc)/(_apert_v/2),
                              1+np.abs(_offst_ffs_y)/(_apert_v/2),1+np.abs(_offst_bfs_y)/(_apert_v/2)])
            ky = k
            #y = np.linspace(-k * _apert_v, k * _apert_v, _ny)
        else:
            y = _axis_y
            axis_y = True

        if _foc_plane == 3:
            if axis_x is False and axis_y is False:
                k = np.amax([kx,ky])
                x = np.linspace(-k * _apert_h, k * _apert_h, _nx)
                y = np.linspace(-k * _apert_v, k * _apert_v, _ny)
        else:
            if axis_x is False and axis_y is False:
                x = np.linspace(-kx * _apert_h, kx * _apert_h, _nx)
                y = np.linspace(-ky * _apert_v, ky * _apert_v, _ny)

        # ========================================================
        # ========== Front Focusing Surface calculations =========
        # ========================================================

        # ------------- new aperture resulting from different penetration depths
        neg_values_ffs_x = False
        neg_values_ffs_y = False

        if _foc_plane == 1 or _foc_plane == 3:
            if _wt_offst_ffs != 0:
                _wall_thick_ffs = _wall_thick + _wt_offst_ffs*2
                _apert_h_ffs = 2*np.sqrt(2*(L_half-_wall_thick_ffs/2)*_r_min)
                if _wall_thick_ffs < 0:
                    neg_values_ffs_x = True
            else:
                _wall_thick_ffs = _wall_thick
                _apert_h_ffs = _apert_h

        elif _foc_plane == 2 or _foc_plane == 3:
            neg_values_ffs_y = False
            if _wt_offst_ffs != 0:
                _wall_thick_ffs = _wall_thick + _wt_offst_ffs*2
                _apert_v_ffs = 2*np.sqrt(2*(L_half-_wall_thick_ffs/2)*_r_min)
                if _wall_thick_ffs < 0:
                    neg_values_ffs_y = True
            else:
                _wall_thick_ffs = _wall_thick
                _apert_v_ffs = _apert_v

        # ------------- calculation of thickness profile in projection approximation

        X, Y = np.meshgrid(x, y)

        mask = np.zeros((_ny, _nx), dtype=bool)

        if _aperture == 'r':
            mask[X - _xc - _offst_ffs_x < -0.5 * _apert_h_ffs] = True
            mask[X - _xc - _offst_ffs_x > 0.5 * _apert_h_ffs] = True
            mask[Y - _yc - _offst_ffs_y < -0.5 * _apert_v_ffs] = True
            mask[Y - _yc - _offst_ffs_y > 0.5 * _apert_v_ffs] = True

        if _aperture == 'c':
            R = ((X - _xc - _offst_ffs_x)**2 + (Y - _yc - _offst_ffs_y) ** 2) ** 0.5
            mask[R > 0.5 * _apert_h_ffs] = True

        delta_z_ffs = (X - _xc - _offst_ffs_x)**2 /_r_min /2 + (Y - _yc - _offst_ffs_y) ** 2 /_r_min /2 + _wall_thick_ffs/2
        delta_z_ffs[mask] = L_half

        if neg_values_ffs_x:
            f_mask = np.zeros(X.shape, dtype=bool)
            f_mask[delta_z_ffs < 0] = True
            delta_z_ffs[f_mask] = 0
        if neg_values_ffs_y:
            f_mask = np.zeros(Y.shape, dtype=bool)
            f_mask[delta_z_ffs < 0] = True
            delta_z_ffs[f_mask] = 0

        # 200728RC: at_rotate_2D() and at_rotate_2D_2steps(): are showing problems. Will use old, but slow function for
        # first release before SPIE2020 and later revisit this issue.

        # ------------- rotation of the front focusing parabolic surface
        if _tilt_ffs_x != 0 or _tilt_ffs_y != 0:
            tilt = np.ones(delta_z_ffs.shape) * L_half
            rz = delta_z_ffs
            rx, ry, rz = at_rotate_2D_nested_loop(x, y, rz, th_x=_tilt_ffs_x, th_y=_tilt_ffs_y, isdgr=isdgr)
            rx, ry, tilt = at_rotate_2D_nested_loop(x, y, tilt, th_x=_tilt_ffs_x, th_y=_tilt_ffs_y, isdgr=isdgr)

            delta_z_ffs = rz - tilt + L_half

        # ------------- full rotation of the CRL_half applied to the full front focusing surface
        if _ang_rot_ex != 0 or _ang_rot_ey != 0:
            tilt = np.ones(delta_z_ffs.shape) * L_half
            offset = np.zeros(delta_z_ffs.shape) * L_half
            rz = delta_z_ffs
            rx, ry, rz = at_rotate_2D_nested_loop(x, y, rz, th_x=_ang_rot_ex, th_y=_ang_rot_ey, isdgr=isdgr)
            if project:
                rx, ry, tilt = at_rotate_2D_nested_loop(x, y, tilt, th_x=_ang_rot_ex, th_y=_ang_rot_ey, isdgr=isdgr)
                rx, ry, offset = at_rotate_2D_nested_loop(x, y, offset, th_x=_ang_rot_ex, th_y=_ang_rot_ey, isdgr=isdgr)

            dc = tilt[int(offset.shape[0]/2),int(offset.shape[1]/2)] + offset[int(offset.shape[0]/2),int(offset.shape[1]/2)]
            delta_z_ffs = rz - tilt + dc

        # ========================================================
        # ========== Back Focusing Surface calculations ==========
        # ========================================================

        if _n == 2:
            # ------------- new aperture resulting from different penetration depths
            neg_values_bfs_x = False
            neg_values_bfs_y = False

            if _foc_plane == 1 or _foc_plane == 3:
                if _wt_offst_bfs != 0:
                    _wall_thick_bfs = _wall_thick + _wt_offst_bfs * 2
                    _apert_h_bfs = 2 * np.sqrt(2 * (L_half - _wall_thick_bfs / 2) * _r_min)
                    if _wall_thick_bfs < 0:
                        neg_values_bfs_x = True
                else:
                    _wall_thick_bfs = _wall_thick
                    _apert_h_bfs = _apert_h

            elif _foc_plane == 2 or _foc_plane == 3:
                neg_values_bfs_y = False
                if _wt_offst_bfs != 0:
                    _wall_thick_bfs = _wall_thick + _wt_offst_bfs * 2
                    _apert_v_bfs = 2 * np.sqrt(2 * (L_half - _wall_thick_bfs / 2) * _r_min)
                    if _wall_thick_bfs < 0:
                        neg_values_bfs_y = True
                else:
                    _wall_thick_bfs = _wall_thick
                    _apert_v_bfs = _apert_v

            # ------------- calculation of thickness profile in projection approximation

            X, Y = np.meshgrid(x, y)

            mask = np.zeros((_ny, _nx), dtype=bool)

            if _aperture == 'r':
                mask[X - _xc - _offst_bfs_x < -0.5 * _apert_h_bfs] = True
                mask[X - _xc - _offst_bfs_x > 0.5 * _apert_h_bfs] = True
                mask[Y - _yc - _offst_bfs_y < -0.5 * _apert_v_bfs] = True
                mask[Y - _yc - _offst_bfs_y > 0.5 * _apert_v_bfs] = True

            if _aperture == 'c':
                R = ((X - _xc - _offst_bfs_x) ** 2 + (Y - _yc - _offst_bfs_y) ** 2) ** 0.5
                mask[R > 0.5 * _apert_h_bfs] = True

            delta_z_bfs = (X - _xc - _offst_bfs_x) ** 2 / _r_min / 2 + (Y - _yc - _offst_bfs_y) ** 2 / _r_min / 2 + _wall_thick_bfs / 2
            delta_z_bfs[mask] = L_half

            if neg_values_bfs_x:
                f_mask = np.zeros(X.shape, dtype=bool)
                f_mask[delta_z_bfs < 0] = True
                delta_z_bfs[f_mask] = 0
            if neg_values_bfs_y:
                f_mask = np.zeros(Y.shape, dtype=bool)
                f_mask[delta_z_bfs < 0] = True
                delta_z_bfs[f_mask] = 0

            # 200728RC: at_rotate_2D() and at_rotate_2D_2steps(): are showing problems. Will use old, but slow function for
            # first release before SPIE2020 and later revisit this issue.

            # ------------- rotation of the front focusing parabolic surface
            if _tilt_bfs_x != 0 or _tilt_bfs_y != 0:
                tilt = np.ones(delta_z_bfs.shape) * L_half
                rz = delta_z_bfs
                rx, ry, rz = at_rotate_2D_nested_loop(x, y, rz, th_x=-_tilt_bfs_x, th_y=-_tilt_bfs_y, isdgr=isdgr)
                rx, ry, tilt = at_rotate_2D_nested_loop(x, y, tilt, th_x=-_tilt_bfs_x, th_y=-_tilt_bfs_y, isdgr=isdgr)

                delta_z_bfs = rz - tilt + L_half

            # ------------- full rotation of the CRL_half applied to the full front focusing surface
            if _ang_rot_ex != 0 or _ang_rot_ey != 0:
                tilt = np.ones(delta_z_bfs.shape) * L_half
                offset = np.zeros(delta_z_bfs.shape) * L_half
                rz = delta_z_bfs
                rx, ry, rz = at_rotate_2D_nested_loop(x, y, rz, th_x=-_ang_rot_ex, th_y=-_ang_rot_ey, isdgr=isdgr)
                if project:
                    rx, ry, tilt = at_rotate_2D_nested_loop(x, y, tilt, th_x=-_ang_rot_ex, th_y=-_ang_rot_ey, isdgr=isdgr)
                    rx, ry, offset = at_rotate_2D_nested_loop(x, y, offset, th_x=-_ang_rot_ex, th_y=-_ang_rot_ey,isdgr=isdgr)

                dc = tilt[int(offset.shape[0] / 2), int(offset.shape[1] / 2)] + offset[
                    int(offset.shape[0] / 2), int(offset.shape[1] / 2)]
                delta_z_bfs = rz - tilt + dc

            return x, y, delta_z_bfs + delta_z_ffs
        return x, y, delta_z_ffs


def polynomial_surface_2D(_z_coeffs, _pol,  _apert_h, _apert_v, _nx=1001, _ny=1001):
    """
    If _z_coeffs is a single number, it refers to the piston value. So an array or random numbers representing the
    coefficients of the polynomials will be generated from -1 to 1 and later normalised to _z_coeffs. If _zcoeffs is a
    list, the function will return a surface based on it.
    :param _z_coeffs: either a list of polynomial coefficients or the total RMS value of the surface errors [m]
    :param _pol: 'c' - circular Zernike; 'r' - rectangular polynomials; 'l' - legendre polynomils
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _nx: number of points vs horizontal position to represent the transmission element
    :param _ny: number of points vs vertical position to represent the transmission element
    :return: thickness profile
    """

    piston = np.nan
    if isinstance(_z_coeffs, Number):
        piston = _z_coeffs
        _z_coeffs = np.zeros(44)
        for i in range(_z_coeffs.size):
            _z_coeffs[i] = random.randrange(-1e6, 1e6) * 1e-6

    npix = [int(_ny/2), int(_nx/2)]

    if _pol is 'c':
        wfr = calc_zernike_circ(_z_coeffs[0:36], npix[0], zern_data={}, mask=True)
    elif _pol is 'r':
        wfr = calc_zernike_rec(_z_coeffs[0:15], npix, zern_data={}, mask=True)
    elif _pol is 'l':
        wfr = calc_legendre(_z_coeffs[0:44], npix, leg_data={}, mask=True)

    if piston is not np.nan:
        if _pol is 'c':
            print('Zernike circle polynomial coefficients:')
            Zcoeffs, fit, residues = fit_zernike_circ(wfr, nmodes=37, startmode=1, rec_zern=False)

        elif _pol is 'r':
            print('Zernike rectangular polynomial coefficients:')
            Zcoeffs, fit, residues = fit_zernike_rec(wfr, nmodes=15, startmode=1, rec_zern=True)

        elif _pol is 'l':
            print('Legendre 2D polynomial coefficients:')
            Zcoeffs, fit, residues = fit_legendre(wfr, nmodes=44, startmode=1, rec_leg=True)
        print(wfr.shape)
        wfr *= piston/Zcoeffs[0]

        Zcoeffs *= piston/Zcoeffs[0]
        k = 1
        coeffslist = ''
        for i in range(Zcoeffs.size):
            if k % 10 is 0:
                coeffslist += 'P' + str(int(k)) + ' = %.2e; \n' % Zcoeffs[i]
            else:
                coeffslist += 'P' + str(int(k)) + ' = %.2e; ' % Zcoeffs[i]
            k += 1
        print(coeffslist)

    x = np.linspace(-_apert_h/2, _apert_h/2, wfr.shape[1])
    y = np.linspace(-_apert_v/2, _apert_v/2, wfr.shape[0])

    return x, y, wfr


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Affine transformation matrices
# ----------------------------------------------------------------------------------------------------------------------

def at_translate(tx=0, ty=0, tz=0):
    """
    Translation matrix.

    [x']   [1  0  0  tx][x]
    [y'] = [0  1  0  ty][y]
    [z']   [0  0  1  tx][z]
    [1 ]   [0  0  0  1 ][1]

    :param tx: translation along the x-axis
    :param ty: translation along the y-axis
    :param tz: translation along the z-axis
    :return: translation matrix
    """

    t = np.identity(4)
    t[0, 3] = tx
    t[1, 3] = ty
    t[2, 3] = tz

    return t


def at_Rx(theta=0, isdgr=False):
    """
    Rotation around the x axis.

    [x']   [1  0   0  0][x]
    [y'] = [0  c  -s  0][y]
    [z']   [0  s   c  0][z]
    [1 ]   [0  0   0  1][1]

    s = sin(theta)
    c = cos(theta)

    :param theta: rotation angle in radians
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :return: rotation matrix
    """
    if isdgr:
        theta *= np.pi / 180

    t = np.zeros([4, 4])
    t[3, 3] = 1

    t[0, 0] = 1
    t[1, 1] = np.cos(theta)
    t[1, 2] = -np.sin(theta)
    t[2, 1] = np.sin(theta)
    t[2, 2] = np.cos(theta)

    return t


def at_Ry(theta=0, isdgr=False):
    """
    Rotation around the y axis.

    [x']   [ c  0  s  0][x]
    [y'] = [ 0  1  0  0][y]
    [z']   [-s  0  c  0][z]
    [1 ]   [ 0  0  0  1][1]

    s = sin(theta)
    c = cos(theta)

    :param theta: rotation angle in radians
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :return: rotation matrix
    """
    if isdgr:
        theta *= np.pi / 180

    t = np.zeros([4, 4])
    t[3, 3] = 1

    t[0, 0] = np.cos(theta)
    t[0, 2] = np.sin(theta)
    t[1, 1] = 1
    t[2, 0] = -np.sin(theta)
    t[2, 2] = np.cos(theta)

    return t


def at_Rz(theta=0, isdgr=False):
    """
    Rotation around the z axis.

    [x']   [c  -s  0  0][x]
    [y'] = [s   c  0  0][y]
    [z']   [0   0  1  0][z]
    [1 ]   [0   0  0  1][1]

    s = sin(theta)
    c = cos(theta)

    :param theta: rotation angle in radians
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :return: rotation matrix
    """
    if isdgr:
        theta *= np.pi / 180

    t = np.zeros([4, 4])
    t[3, 3] = 1

    t[0, 0] = np.cos(theta)
    t[0, 1] = -np.sin(theta)
    t[1, 0] = np.sin(theta)
    t[1, 1] = np.cos(theta)
    t[2, 2] = 1

    return t

def at_apply(R, x, y, z):
    """
    Apply a transformation matrix R to a set of cardinal points (x,y,z)
    :param R: transformation matrix
    :param x: x-coordinates
    :param y: y-coordinates
    :param z: z-coordinates
    :return: transformed coordinates
    """
    xp = x * R[0, 0] + y * R[0, 1] + z * R[0, 2]
    yp = x * R[1, 0] + y * R[1, 1] + z * R[1, 2]
    zp = x * R[2, 0] + y * R[2, 1] + z * R[2, 2]
    return xp, yp, zp


def at_rotate_1D(x, f_x, th=0, isdgr=False, project=False):
    """
    Rotates a set of points f_x(x) a angle of th.
    :param x: abscissa coordinates
    :param f_x: ordinate values
    :param th: rotation angle
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :param project: boolean for recalculating the rotated profile on the original abscissa coordinates
    :return: rotated coordinates pairs (x,f_x)
    """
    try:
        y = np.zeros(x.shape)
    except:
        y = 0
    R = at_Ry(th, isdgr)
    xp, yp, zp = at_apply(R, x, y, f_x)

    if project:
        f = interp1d(xp, zp, bounds_error=False, fill_value=0)
        t_profile = f(x)
        return x, t_profile
    else:
        return xp, zp


def at_rotate_2D_nested_loop(x, y, z, th_x=0, th_y=0, isdgr=False):
    '''
    Rotates a cloud point z(x,y) around theta_x and then, theta_y.
    :param x: horizontal axis
    :param y: vertical axis
    :param z: cloud point to be rotate: z(x,y)
    :param th_x: angle around the x-axis for the rotation
    :param th_y: angle around the y-axis for the rotation
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :return: rotated coordinates pairs
    '''

    tilted_image = np.zeros(z.shape)

    # applying RX:
    if th_x != 0:
        for i in range(x.size):
            cut = z[:,i]
            rx, tilted_image[:,i] = at_rotate_1D(y, cut, th=th_x, isdgr=isdgr, project=True)
    else:
        tilted_image = z

    if th_y != 0:
        for i in range(y.size):
            cut = tilted_image[i,:]
            rx, tilted_image[i,:] = at_rotate_1D(x, cut, th=th_y, isdgr=isdgr, project=True)

    return x, y, tilted_image


# 200728RC: revisit functions that are acting up
def at_rotate_2D(x, y, f_x, th_x=0, th_y=0, isdgr=False, project=False):
    '''
    Rotates a cloud point z(x,y) around theta_x and then, theta_y. 200728RC ATTENTION: function is not working
    :param x: horizontal axis
    :param y: vertical axis
    :param z: cloud point to be rotate: z(x,y)
    :param th_x: angle around the x-axis for the rotation
    :param th_y: angle around the y-axis for the rotation
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :param project: boolean for recalculating the rotated profile on the original grid
    :return: rotated coordinates pairs
    '''
    Ry = at_Ry(th_y, isdgr)
    Rx = at_Rx(th_x, isdgr)
    R = np.matmul(Ry, Rx)
    xp, yp, zp = at_apply(R, x, y, f_x)

    if project:
        f = interp2d(xp[0, :], yp[:, 0], zp, kind='linear', bounds_error=False, fill_value=0)
        t_profile = f(x[0, :], y[:, 0])
        return x[0, :], y[:, 0], t_profile
    else:
        return xp[0, :], yp[:, 0], zp


def at_rotate_2D_2steps(x, y, z, th_x=0, th_y=0, isdgr=False, project=False):
    '''
     Rotates a cloud point z(x,y) around theta_x and then, theta_y. 200728RC ATTENTION: function is not working
    :param x: horizontal axis
    :param y: vertical axis
    :param z: cloud point to be rotate: z(x,y)
    :param th_x: angle around the x-axis for the rotation
    :param th_y: angle around the y-axis for the rotation
    :param isdgr: boolean for determining if angle is in degree or in radians (default)
    :param project: boolean for recalculating the rotated profile on the original grid
    :return: rotated coordinates pairs
    '''
    Ry = at_Ry(th_y, isdgr)
    Rx = at_Rx(th_x, isdgr)

    # applying RX:
    if th_x != 0:
        xp, yp, zp = at_apply(Rx, x, y, z)
        if project:
            f = interp2d(xp[0, :], yp[:, 0], zp, kind='linear', bounds_error=False, fill_value=0)
            t_profile = f(x[0, :], y[:, 0])
            zp = t_profile
    else:
        zp = z

    # applying RY:
    if th_y != 0:
        xp, yp, zp = at_apply(Ry, x, y, zp)
        if project:
            f = interp2d(xp[0, :], yp[:, 0], zp, kind='linear', bounds_error=False, fill_value=0)
            t_profile = f(x[0, :], y[:, 0])
            zp = t_profile

    return x[0, :], y[:, 0], zp

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
