#!/usr/bin/python
# coding: utf-8

###################################################################################
# barc4RefractiveOptics
# Authors/Contributors: Rafael Celestre, Oleg Chubar, Manuel Sanchez del Rio
# Rafael.Celestre@esrf.eu
# creation: 24.06.2019
# last update: 06.01.2023 (v0.4)
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
from numpy import fft
from numbers import Number
import random

from barc4ro.wavefront_fitting import *
from barc4ro.barc4utils import *

from copy import deepcopy


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Projected thickness calculation
# ----------------------------------------------------------------------------------------------------------------------

def proj_thick_1D_crl(*args, **kwargs):  # this is for back-compatibility
    return proj_thick_crl_1D(*args, **kwargs)

def proj_thick_crl_1D(_shape, _apert_h, _r_min, _n=2, _wall_thick=0, _xc=0, _nx=1001, _ang_rot_ex=0, _offst_ffs_x=0,
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


def proj_thick_2D_crl(*args, **kwargs):  # this is for back-compatibility
    return proj_thick_crl_2D(*args, **kwargs)

def proj_thick_crl_2D(_foc_plane, _shape, _apert_h, _apert_v, _r_min, _n, _wall_thick=0, _xc=0, _yc=0, _nx=1001,
                      _ny=1001, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0, _offst_ffs_x=0, _offst_ffs_y=0,
                      _tilt_ffs_x=0, _tilt_ffs_y=0, _ang_rot_ez_ffs=0, _wt_offst_ffs=0, _offst_bfs_x=0, _offst_bfs_y=0,
                      _tilt_bfs_x=0, _tilt_bfs_y=0, _ang_rot_ez_bfs=0, _wt_offst_bfs=0, isdgr=False, project=True,
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
            alpha_max_y = (np.pi/2 - np.arctan(_apert_h/2/_r_min)) * 180/np.pi
            alpha_max_x = (np.pi/2 - np.arctan(_apert_v/2/_r_min)) * 180/np.pi

        else:
             alpha_max_y = (np.pi/2 - np.arctan(_apert_h/2/_r_min))
             alpha_max_x = (np.pi/2 - np.arctan(_apert_v/2/_r_min))

        if _ang_rot_ex>0.98*alpha_max_x or _tilt_ffs_x>0.98*alpha_max_x or _tilt_bfs_x>0.98*alpha_max_x:
            print('>>> Tilt angle causes obscuration in the vertical direction - error in interpolation'
                  ' on the edges of the lens may be present')

        if _ang_rot_ey>0.98*alpha_max_y or _tilt_ffs_y>0.98*alpha_max_y or _tilt_bfs_y>0.98*alpha_max_y:
            print('>>> Tilt angle causes obscuration in the horizontal direction - error in interpolation'
                  ' on the edges of the lens may be present')

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

        _apert_h_ffs = _apert_h
        _apert_v_ffs = _apert_v

        if _foc_plane == 1 or _foc_plane == 3:
            if _wt_offst_ffs != 0:
                _wall_thick_ffs = _wall_thick + _wt_offst_ffs*2
                _apert_h_ffs = 2*np.sqrt(2*(L_half-_wall_thick_ffs/2)*_r_min)
                if _wall_thick_ffs < 0:
                    neg_values_ffs_x = True
            else:
                _wall_thick_ffs = _wall_thick

        if _foc_plane == 2 or _foc_plane == 3:
            neg_values_ffs_y = False
            if _wt_offst_ffs != 0:
                _wall_thick_ffs = _wall_thick + _wt_offst_ffs*2
                _apert_v_ffs = 2*np.sqrt(2*(L_half-_wall_thick_ffs/2)*_r_min)
                if _wall_thick_ffs < 0:
                    neg_values_ffs_y = True
            else:
                _wall_thick_ffs = _wall_thick

        # ------------- calculation of thickness profile in projection approximation

        # X, Y = np.meshgrid(x, y)
        # mask = np.zeros((_ny, _nx), dtype=bool)

        X = np.outer(np.ones_like(y), x)
        Y = np.outer(y, np.ones_like(x))
        mask = np.zeros((_ny, _nx), dtype=bool)

        if _aperture == 'r':
            mask[X - _xc - _offst_ffs_x < -0.5 * _apert_h_ffs] = True
            mask[X - _xc - _offst_ffs_x > 0.5 * _apert_h_ffs] = True
            mask[Y - _yc - _offst_ffs_y < -0.5 * _apert_v_ffs] = True
            mask[Y - _yc - _offst_ffs_y > 0.5 * _apert_v_ffs] = True

        if _aperture == 'c':
            R = ((X - _xc - _offst_ffs_x)**2 + (Y - _yc - _offst_ffs_y) ** 2) ** 0.5
            mask[R > 0.5 * _apert_h_ffs] = True

        # delta_z_ffs = (X - _xc - _offst_ffs_x)**2 /_r_min /2 + (Y - _yc - _offst_ffs_y) ** 2 /_r_min /2 + _wall_thick_ffs/2
        #  :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
        if _foc_plane == 3:
            delta_z_ffs = (X - _xc - _offst_ffs_x) ** 2 / _r_min / 2 + (Y - _yc - _offst_ffs_y) ** 2 / _r_min / 2 + _wall_thick_ffs / 2
        elif _foc_plane == 2:
            delta_z_ffs = (Y - _yc - _offst_ffs_y) ** 2 / _r_min / 2 + _wall_thick_ffs / 2
        elif _foc_plane == 1:
            delta_z_ffs = (X - _xc - _offst_ffs_x) ** 2 / _r_min / 2 + _wall_thick_ffs / 2


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
            _apert_h_bfs = _apert_h
            _apert_v_bfs = _apert_v

            if _foc_plane == 1 or _foc_plane == 3:
                if _wt_offst_bfs != 0:
                    _wall_thick_bfs = _wall_thick + _wt_offst_bfs * 2
                    _apert_h_bfs = 2 * np.sqrt(2 * (L_half - _wall_thick_bfs / 2) * _r_min)
                    if _wall_thick_bfs < 0:
                        neg_values_bfs_x = True
                else:
                    _wall_thick_bfs = _wall_thick

            if _foc_plane == 2 or _foc_plane == 3:
                neg_values_bfs_y = False
                if _wt_offst_bfs != 0:
                    _wall_thick_bfs = _wall_thick + _wt_offst_bfs * 2
                    _apert_v_bfs = 2 * np.sqrt(2 * (L_half - _wall_thick_bfs / 2) * _r_min)
                    if _wall_thick_bfs < 0:
                        neg_values_bfs_y = True
                else:
                    _wall_thick_bfs = _wall_thick

            # ------------- calculation of thickness profile in projection approximation

            # X, Y = np.meshgrid(x, y)
            # mask = np.zeros((_ny, _nx), dtype=bool)

            X = np.outer(np.ones_like(y), x)
            Y = np.outer(y, np.ones_like(x))
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
            #  :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
            if _foc_plane == 3:
                delta_z_bfs = (X - _xc - _offst_bfs_x) ** 2 / _r_min / 2 + (
                            Y - _yc - _offst_bfs_y) ** 2 / _r_min / 2 + _wall_thick_bfs / 2
            elif _foc_plane == 2:
                delta_z_bfs = (Y - _yc - _offst_bfs_y) ** 2 / _r_min / 2 + _wall_thick_bfs / 2
            elif _foc_plane == 1:
                delta_z_bfs = (X - _xc - _offst_bfs_x) ** 2 / _r_min / 2 + _wall_thick_bfs / 2


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


def proj_thick_axicon_2D(_foc_plane, _shape, _apert_h, _apert_v, _h, _wall_thick=0, _xc=0, _yc=0, _nx=1001, _ny=1001,
                        _axis_x=None, _axis_y=None):
    """

    :param _foc_plane: plane of focusing: 1- horizontal, 2- vertical, 3- both
    :param _shape: 'p' - positive or 'n' - negative
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _h: height of the axicon tip/depression (always positive) [m]
    :param _wall_thick:  support/substrate thickness [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _nx: number of points vs horizontal position to represent the transmission element
    :param _ny: number of points vs vertical position to represent the transmission element
    :param _axis_x: forces the lens to be calculated on a given grid - avoids having to interpolate different calculations to the same grid
    :param _axis_y: forces the lens to be calculated on a given grid - avoids having to interpolate different calculations to the same grid
    :return: thickness profile

    """

    k = 0.7

    if _axis_x is None:
        _axis_x = np.linspace(-k * _apert_h, k * _apert_h, _nx)

    if _axis_y is None:
        _axis_y = np.linspace(-k * _apert_v, k * _apert_v, _ny)

    X, Y = np.meshgrid(_axis_x, _axis_y)

    delta_z =  np.zeros((_ny, _nx))

    if _foc_plane == 1:
        ls =  2*_h/_apert_h*(X-_xc) + _h
        rs = -2*_h/_apert_h*(X-_xc) + _h
        ls[X-_xc>0] = 0
        rs[X-_xc<=0] = 0

    elif _foc_plane == 2:
        ls =  2*_h/_apert_v*(Y-_yc) + _h
        rs = -2*_h/_apert_v*(Y-_yc) + _h
        ls[Y-_yc>0] = 0
        rs[Y-_yc<=0] = 0

    if _foc_plane == 3:

        R = np.sqrt((X-_xc)**2 +(Y-_yc)**2)
        delta_z = -2*_h/_apert_h*R + _h
        delta_z[R>_apert_h/2] = 0

    else:
        delta_z = ls+rs
        mask = np.zeros((_ny, _nx), dtype=bool)
        mask[X-_xc < -0.5 * _apert_h] = True
        mask[X-_xc > 0.5  * _apert_h] = True
        mask[Y-_yc < -0.5 * _apert_v] = True
        mask[Y-_yc > 0.5  * _apert_v] = True
        delta_z[mask] = 0

    if _shape == 'n':
        delta_z *=-1
        delta_z -= np.amin(delta_z)

    return _axis_x, _axis_y, (delta_z+_wall_thick)


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

    if _pol == 'c':
        wfr = calc_zernike_circ(_z_coeffs[0:36], npix[0], zern_data={}, mask=True)
    elif _pol == 'r':
        wfr = calc_zernike_rec(_z_coeffs[0:15], npix, zern_data={}, mask=True)
    elif _pol == 'l':
        wfr = calc_legendre(_z_coeffs[0:44], npix, leg_data={}, mask=True)

    if piston is not np.nan:
        if _pol == 'c':
            print('Zernike circle polynomial coefficients:')
            Zcoeffs, fit, residues = fit_zernike_circ(wfr, nmodes=37, startmode=1, rec_zern=False)

        elif _pol == 'r':
            print('Zernike rectangular polynomial coefficients:')
            Zcoeffs, fit, residues = fit_zernike_rec(wfr, nmodes=15, startmode=1, rec_zern=True)

        elif _pol == 'l':
            print('Legendre 2D polynomial coefficients:')
            Zcoeffs, fit, residues = fit_legendre(wfr, nmodes=44, startmode=1, rec_leg=True)
        print(wfr.shape)
        wfr *= piston/Zcoeffs[0]

        Zcoeffs *= piston/Zcoeffs[0]
        k = 1
        coeffslist = ''
        for i in range(Zcoeffs.size):
            if k % 10 == 0:
                coeffslist += 'P' + str(int(k)) + ' = %.2e; \n' % Zcoeffs[i]
            else:
                coeffslist += 'P' + str(int(k)) + ' = %.2e; ' % Zcoeffs[i]
            k += 1
        print(coeffslist)

    x = np.linspace(-_apert_h/2, _apert_h/2, wfr.shape[1])
    y = np.linspace(-_apert_v/2, _apert_v/2, wfr.shape[0])

    return x, y, wfr


def fractal_surf(_sigma, _psd_slope, _pix_size, _m, _n, _qr=0, _dist=0, _seed=None, _psd=False, _C=None):
    """
    Generates a 2D random (rough) surface in [m] with a pre-determined PSD. The PSD can be defined by either the rms
    value of the roughness (_sigma), _psd_slope and roll-off freq. (_qr); or by a directly calculated 2D PSD (_C). A
    random phase is added to the PSD in order to generate the surface. The random distribution can be uniform (_dist=0),
    Gaussian (_dist=1) or even zero (_dist=-1). A _seed can be given to the random generator. If _psd is True, the
    2D PSD and its axes are also returned.

    parameters (in SI units)

    :param _sigma: root-mean-square roughness Rq(m)
    :param _psd_slope: PSD _psd_slope = -2(H+1); Hurst _psd_slope 0<= H <= 1, fractal dimension D = 3-H
    :param _pix_size: pixel size in [m] for the resulting surface
    :param _m: number of pixels in x
    :param _n: number of pixels in y
    :param _qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/_pix_size) - Nyquist freq.
    :param _dist: -1 for phase = 0, 0 for uniform phase distribution, 1 for Gaussian dist.
    :param _seed: seed for random initialisation
    :param _psd: (bool) if true, returns Cq and its vectors
    :param _C: pre-calculated 2D psd where qx and qy respect the limits imposed by _pix_size, _m and _n
    :return: surface profile and axes

    % Fractal topographies with different fractal dimensions.
    Adaptation of the MATLAB function 'artificial_surf' (version 1.1.0.0) by Mona Mahboob Kanafi.
    https://www.mathworks.com/matlabcentral/fileexchange/60817-surface-generator-artificial-randomly-rough-surfaces
    """

    # =========================================================================
    # 2D matrix of Cq (PSD) values
    if _C is None:
        _C, qx, qy = uti_gen_2d_psd_from_param(_sigma, _psd_slope, _pix_size, _m, _n, _qr=_qr)
    else:
        qx = fft.fftshift(fft.fftfreq(_m, _pix_size))
        qy = fft.fftshift(fft.fftfreq(_n, _pix_size))

    z, y, x = fractal_surf_from_2d_psd(_C, _sigma, _pix_size, _dist, _seed)

    if _psd:
        return z, y, x, _C, qy, qx
    else:
        return z, y, x


def fractal_surf_from_2d_psd(_C, _sigma, _pix_size, _dist=0, _seed=None):
    """
    Generates a 2D random (rough) surface in [m] with a pre-determined 2D PSD (_C). The surface roughness (rms) is
    imposed by _sigma. A random phase is added to the PSD in order to generate the surface. The random distribution can
    be uniform (_dist=0), Gaussian (_dist=1) or even zero (_dist=-1). A _seed can be given to the random generator.

    parameters (in SI units)

    :param _C: 2D PSD
    :param _sigma: root-mean-square roughness Rq(m)
    :param _pix_size: pixel size in [m] for the resulting surface
    :param _dist: -1 for phase = 0, 0 for uniform phase distribution, 1 for Gaussian dist.
    :param _seed: seed for random initialisation
    :return: surface profile and axes
    """

    n = _C.shape[0]
    m = _C.shape[1]
    psd = deepcopy(_C)
    psd *= 2    # RC - 13Feb2023 - correction factor
    # =========================================================================
    # applying rms
    # psd *= (_sigma / (np.sqrt(np.sum(psd) / (m * _pix_size * n * _pix_size)))) ** 2

    # =========================================================================
    # reversing operation: PSD to fft
    # Bq = np.sqrt(psd / (_pix_size ** 2 / (n * m)))
    Bq = np.sqrt(psd * (n * m) / (_pix_size ** 2))

    # =========================================================================
    # defining a random phase
    np.random.seed(_seed)
    if _dist == -1:
        phi = np.zeros((n, m))
    elif _dist == 0:
        phi = -np.pi + 2 * np.pi * np.random.uniform(0, 1, (n, m))
    elif _dist == 1:
        phi = np.pi * np.random.normal(0, 1, (n, m))
    # =========================================================================
    #  generates surface
    z = np.abs(fft.ifftshift(fft.ifft2(Bq * np.exp(-1j * phi))))
    z -= calc_surf_rms(z)
    z -= np.mean(z)
    z += _sigma
    z[int(n / 2), int(m / 2)] = np.median(z[int(n / 2)-5:int(n / 2)+5, int(m / 2)-5:int(m / 2)+5])
    x = np.linspace(-m / 2, m / 2, m) * _pix_size
    y = np.linspace(-n / 2, n / 2, n) * _pix_size
    return z, y, x


def fractal_surf_from_psd_param(_sigma, _psd_slope, _pix_size, _m, _n, _qr=0, _dist=0, _seed=None):
    """
    Generates a 2D random (rough) surface in [m] with a pre-determined PSD defined by the rms value of the roughness
    (_sigma), _psd_slope and roll-off freq. (_qr); A random phase is added to the PSD in order to generate the surface.
    The random distribution can be uniform (_dist=0), Gaussian (_dist=1) or even zero (_dist=-1). A _seed can be given
    to the random generator.
    parameters (in SI units)

    :param _sigma: root-mean-square roughness Rq(m)
    :param _psd_slope: PSD _psd_slope = -2(H+1); Hurst _psd_slope 0<= H <= 1, fractal dimension D = 3-H
    :param _pix_size: pixel size in [m] for the resulting surface
    :param _m: number of pixels in x
    :param _n: number of pixels in y
    :param _qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/_pix_size) - Nyquist freq.
    :param _dist: -1 for phase = 0, 0 for uniform phase distribution, 1 for Gaussian dist.
    :param _seed: seed for random initialisation
    :return: surface profile and axes

    """

    Cq, qx, qy = uti_gen_2d_psd_from_param(_sigma, _psd_slope, _pix_size, _m, _n, _qr=_qr)
    z, y, x = fractal_surf_from_2d_psd(Cq, _pix_size, _dist, _seed)

    return z, y, x


def bumps_and_holes_surf(n_bmp, R, x, y, bmp_type=0, xo=None, yo=None, dist_R=0, dist_xo=0, dist_yo=0, seed=69):
    """

    :param n_bmp: number of bumps. Must be >= 1
    :param R: bump or hole main parameters:
                Gaussian bump: R = [amp, amp_bis, sigma_x, sigma_x_bis, sigma_y, sigma_y_bis];
                spherical bump: R = [R, R_bis]
                sinusoid bump: R = [amp, amp_bis, f_x, f_x_bis, f_y, f_y_bis, phase, phase_bis, offset, offset_bis]
                hole: R = [R, R_bis, depth, depth_bis]
                through hole: R = [R, R_bis, depth]

                (value, value_bis) can be (_min, _max) or (_mean, _std) depending on the dist type

    :param x: horizontal axis in [m] (1D array)
    :param y: vertical axis in [m] (1D array)
    :param bmp_type: type of bump or hole
                bmp_type = 0    # Gaussian bump
                bmp_type = 1    # spherical_bump
                bmp_type = 2    # circular holes with different depths
                bmp_type = 3    # circular through holes
                bmp_type = 4    # 2D sinusoid
    :param xo: bumps or holes positions
    :param yo: bumps or holes positions
    :param dist_R: type of distribution for R parameters
                dist = 0        # uniform distribution; requires _min and _max
                dist = 1        # Gaussian distribution; requires _mean and _std
    :param dist_xo: type of distribution for xo parameters
    :param dist_yo: type of distribution for yo parameters
    :param seed: seed for random generators
    :return:
    """

    np.random.seed(seed)

    # -----------------------------------------
    # Distributions for bumps or holes
    if bmp_type == 0:  # Gaussian bump
        if len(R) > 6:
            print('List of Gaussian parameters given')
            list_R = R
        else:
            if dist_R == 0:
                list_amp = np.random.uniform(R[0], R[1], n_bmp)
                list_sigx = np.random.uniform(R[2], R[3], n_bmp)
                list_sigy = np.random.uniform(R[4], R[5], n_bmp)
            elif dist_R == 1:
                list_amp = np.random.normal(R[0], R[1], n_bmp)
                list_sigx = np.random.normal(R[2], R[3], n_bmp)
                list_sigy = np.random.normal(R[4], R[5], n_bmp)
                list_amp[list_amp < 0] = 0
                list_sigx[list_sigx < 0] = 0
                list_sigy[list_sigy < 0] = 0

    elif bmp_type == 1:  # spherical_bump
        if len(R) > 2:
            print('List of radii given')
            list_R = R
        else:
            if dist_R == 0:
                list_R = np.random.uniform(R[0], R[1], n_bmp)
            elif dist_R == 1:
                list_R = np.random.normal(R[0], R[1], n_bmp)
                list_R[list_R < 0] = 0

    elif bmp_type == 2:  # circular holes with different depths
        if len(R) > 4:
            print('List of radii given')
            list_R = R
        else:
            if dist_R == 0:
                list_R = np.random.uniform(R[0], R[1], n_bmp)
                list_d = np.random.uniform(R[2], R[3], n_bmp)
            elif dist_R == 1:
                list_R = np.random.normal(R[0], R[1], n_bmp)
                list_d = np.random.normal(R[2], R[3], n_bmp)
                list_R[list_R < 0] = 0
                list_d[list_d < 0] = 0

    elif bmp_type == 3:  # circular thorugh holes
        if len(R) > 3:
            print('List of radii given')
            list_R = R
        else:
            if dist_R == 0:
                list_R = np.random.uniform(R[0], R[1], n_bmp)
            elif dist_R == 1:
                list_R = np.random.normal(R[0], R[1], n_bmp)
                list_R[list_R < 0] = 0
            depth = R[2]

    elif bmp_type == 4:  # sinusoid
        if len(R) > 10:
            print('List of sinusoid parameters given')
            list_R = R
        else:
            if dist_R == 0:
                list_amp = np.random.uniform(R[0], R[1], n_bmp)
                list_f_x = np.random.uniform(R[2], R[3], n_bmp)
                list_f_y = np.random.uniform(R[4], R[5], n_bmp)
                list_phase = np.random.uniform(R[6], R[7], n_bmp)
                list_offset = np.random.uniform(R[8], R[9], n_bmp)
            elif dist_R == 1:
                list_amp = np.random.uniform(R[0], R[1], n_bmp)
                list_f_x = np.random.uniform(R[2], R[3], n_bmp)
                list_f_y = np.random.uniform(R[4], R[5], n_bmp)
                list_phase = np.random.uniform(R[6], R[7], n_bmp)
                list_offset = np.random.uniform(R[8], R[9], n_bmp)

                # -----------------------------------------
    # Positions for bumps or holes centres
    X = np.outer(np.ones_like(y), x)
    Y = np.outer(y, np.ones_like(x))

    if xo is None:
        list_xo = np.random.uniform(X[0, 0], X[0, -1], n_bmp)
    else:
        if len(xo) > 2:
            print('List of xo given')
            list_xo = xo
        else:
            if dist_xo == 0:
                list_xo = np.random.uniform(xo[0], xo[1], n_bmp)
            elif dist_xo == 1:
                list_xo = np.random.normal(xo[0], xo[1], n_bmp)
    if yo is None:
        list_yo = np.random.uniform(Y[0, 0], Y[-1, 0], n_bmp)
    else:
        if len(yo) > 2:
            print('List of yo given')
            list_yo = yo
        else:
            if dist_yo == 0:
                list_yo = np.random.uniform(yo[0], yo[1], n_bmp)
            elif dist_yo == 1:
                list_yo = np.random.normal(yo[0], yo[1], n_bmp)

    # -----------------------------------------
    # Surface calculation
    surf = 0
    if bmp_type == 3:
        surf = circular_through_holes(depth, list_R, X, Y, list_xo, list_yo)
    else:
        for bmp in range(n_bmp):
            if bmp_type == 0:
                surf += gaussian_bump(list_amp[bmp], list_sigx[bmp], list_sigy[bmp], X, Y, list_xo[bmp], list_yo[bmp])
            elif bmp_type == 1:
                surf += spherical_bump(list_R[bmp], X, Y, list_xo[bmp], list_yo[bmp])
            elif bmp_type == 2:
                surf += circular_holes(list_d[bmp], list_R[bmp], X, Y, list_xo[bmp], list_yo[bmp])
            elif bmp_type == 4:
                surf += sinusoid_bump(list_amp[bmp], list_f_x[bmp], list_f_y[bmp], list_phase[bmp], list_offset[bmp], X,
                                      Y, list_xo[bmp], list_yo[bmp])

    return surf


def spherical_bump(R, X, Y, xo=0, yo=0):
    """

    :param R:
    :param X:
    :param Y:
    :param xo:
    :param yo:
    :return:
    """
    argument = - (X - xo) ** 2 - (Y - yo) ** 2 + R ** 2
    argument[argument < 0] = 0
    return 2 * np.sqrt(argument)


def gaussian_bump(b_amp, sigma_x, sigma_y, X, Y, xo=0, yo=0):
    """

    :param b_amp:
    :param sigma_x:
    :param sigma_y:
    :param X:
    :param Y:
    :param xo:
    :param yo:
    :return:
    """
    return b_amp * np.exp(-0.5 * ((X - xo) / (sigma_x)) ** 2) * np.exp(-0.5 * ((Y - yo) / (sigma_y)) ** 2)


def sinusoid_bump(amp, f_x, f_y, phase, offset, X, Y, xo=0, yo=0):
    """

    :param amp:
    :param f_x:
    :param f_y:
    :param phase:
    :param offset:
    :param X:
    :param Y:
    :param xo:
    :param yo:
    :return:
    """
    return amp * np.sin(2 * np.pi * f_x * (X - xo) + 2 * np.pi * f_y * (Y - yo) + phase) + offset


def circular_holes(depth, R, X, Y, xo=0, yo=0):
    """

    :param depth:
    :param R:
    :param X:
    :param Y:
    :param xo:
    :param yo:
    :return:
    """
    argument = - (X - xo) ** 2 - (Y - yo) ** 2 + R ** 2
    mask = argument < 0
    argument[mask] = depth
    argument[np.logical_not(mask)] = 0
    return argument


def circular_through_holes(depth, R_list, X, Y, xo_list, yo_list):
    """

    :param depth:
    :param R_list:
    :param X:
    :param Y:
    :param xo_list:
    :param yo_list:
    :return:
    """
    mask = np.ones(X.shape, dtype='bool')
    surf = np.ones(X.shape) * depth

    for hole in range(len(R_list)):
        argument = - (X - xo_list[hole]) ** 2 - (Y - yo_list[hole]) ** 2 + R_list[hole] ** 2
        mask = np.logical_and(mask, argument < 0)
    surf[np.logical_not(mask)] = 0

    return surf


# TODO: (RC2023SEP05)
# def fill_ROI_with_pattern(pattern):
#     pass