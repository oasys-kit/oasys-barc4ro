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

from barc4ro.wavefront_fitting import *

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
                      _ny=1001, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0, _offst_ffs_x=0, _offst_ffs_y=0,
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

        X = np.outer(x, np.ones_like(y))
        Y = np.outer(np.ones_like(x), y)
        mask = np.zeros((_nx, _ny), dtype=bool)

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

            X = np.outer(x, np.ones_like(y))
            Y = np.outer(np.ones_like(x), y)
            mask = np.zeros((_nx, _ny), dtype=bool)


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

