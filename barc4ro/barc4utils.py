#!/usr/bin/python
# coding: utf-8

###################################################################################
# barc4RefractiveOptics
# Authors/Contributors: Rafael Celestre, Oleg Chubar, Manuel Sanchez del Rio
# Rafael.Celestre@esrf.eu
# creation: 24.06.2019
# last update: 06.01.2020 (v0.5)
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
from numpy import fft
from scipy.interpolate import interp1d, interp2d


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- PSD generation
# ----------------------------------------------------------------------------------------------------------------------

def uti_gen_2d_psd_from_param(_sigma, _psd_slope, _pix_size, _m, _n, _qr=0, _noise=False, _dist=0, _seed=None, _isfreq=True):
    """
    Generates a 2D PSD can be defined by the rms value of the roughness of a surface (_sigma), _psd_slope and
    roll-off freq. (_qr); A random phase can be added to the PSD in order to generate some high frequency noise if
    _noise=True. The random distribution can be uniform (_dist=0), Gaussian (_dist=1) or even zero (_dist=-1).
    A _seed can be given to the random generator.

    parameters (in SI units)

    :param _sigma: root-mean-square roughness Rq(m)
    :param _psd_slope: PSD _psd_slope = -2(H+1); Hurst _psd_slope 0<= H <= 1, fractal dimension D = 3-H
    :param _pix_size: pixel size. If in [m], set _isfreq = False
    :param _m: number of pixels in x
    :param _n: number of pixels in y
    :param _qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/_pix_size) - Nyquist freq.
    :param _noise: adds some random noise to the ideal PSD.
    :param _dist: -1 for phase = 0, 0 for uniform phase distribution, 1 for Gaussian dist.
    :param _seed: seed for random initialisaiton
    :param _isfreq: boolean defining in _pix_size is in [m] or in [1/m]
    :return: surface profile and axes

    """

    if _isfreq is True:
        x = fft.fftshift(fft.fftfreq(_m, _pix_size))
        _pix_size = x[1]-x[0]

    # =========================================================================
    qx = fft.fftshift(fft.fftfreq(_m, _pix_size))  # image frequency in fx direction
    qy = fft.fftshift(fft.fftfreq(_n, _pix_size))  # image frequency in fy direction

    Qx, Qy = np.meshgrid(qx, qy)

    # cylindrical coordinates in frequency-space
    rho = np.sqrt(Qy ** 2 + Qx ** 2)

    [y0, x0] = np.where(rho == 0)
    rho[y0, x0] = 1  # avoids division by zero

    # 2D matrix of Cq (PSD) values
    Cq = rho ** _psd_slope
    if _qr != 0:
        Cq[np.where(rho < _qr)] = _qr ** _psd_slope

    Cq[y0, x0] = 0

    # =========================================================================
    # applying rms
    Lx = _m * _pix_size  # image length in x direction
    Ly = _n * _pix_size  # image length in y direction
    A = Lx * Ly

    Cq *= (_sigma / (np.sqrt(np.sum(Cq) / A))) ** 2

    if _noise is True:
        # =========================================================================
        # reversing operation: PSD to fft
        Bq = np.sqrt(Cq / (_pix_size ** 2 / (_n * _m)))
        # =========================================================================
        # defining a random phase
        np.random.seed(_seed)
        if _dist == -1:
            phi = np.zeros((_n, _m))
        elif _dist == 0:
            phi = -np.pi + 2 * np.pi * np.random.uniform(0, 1, (_n, _m))
        elif _dist == 1:
            phi = np.pi * np.random.normal(0, 1, (_n, _m))
        # =========================================================================
        # Complex FFT
        Hm = Bq * np.exp(-1j * phi)
        # =========================================================================
        # PSD recalculation
        Cq = (_pix_size**2/(_n * _m))*np.absolute(Hm) ** 2

    return Cq, qy, qx


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- PSD calculation
# ----------------------------------------------------------------------------------------------------------------------

def uti_calc_1d_psd(_profile1D, _axis, _positive_side=False, _pad=False):

    """
    This function receives a height profile cut in [m] as a 1D array (_profile1D) and a coordinate array (_axis) also in
    [m]. A 1D PSD and it's accompanying axis is returned. If _positive_side is True, negative frequencies are returned.
    Padding can be done to increase the frequency resolution.

    :param _profile1D: height profile as a 1D numpy array in [m]
    :param _axis: 1D numpy array in [m]
    :param _positive_side: (boolean) crops all negative frequencies
    :param _pad: (boolean) zero padding to increase the frequency sampling
    :return: 1D psd and the frequency axis
    """

    pix_size = _axis[1] - _axis[0]
    if _pad:
        _profile1D = np.pad(_profile1D, (int(len(_profile1D) / 2), int(len(_profile1D) / 2)))

    m = len(_profile1D)
    freq = fft.fftshift(fft.fftfreq(m, pix_size))
    psd = (pix_size / m) * np.absolute(fft.fftshift(fft.fft(_profile1D))) ** 2

    if _positive_side:
        psd = 2 * psd[freq > 0]
        freq = freq[freq > 0]
    return psd, freq


def uti_calc_2d_psd(_profile2D, _axis_x, _axis_y, _pad=False):

    """
    This function receives a 2D height profile in [m] as numpy array (_profile2D) and two coordinate arrays (_axis_x and
    _axis_y) also in [m]. A 2D PSD and its axes (freqx and freqy) are returned.

    :param _profile2D: height profile as a 2D numpy array in [m]
    :param _axis_x: 1D numpy array in [m]
    :param _axis_y: 1D numpy array in [m]
    :param _pad: (boolean) zero padding to increase the frequency sampling
    :return: 2D psd and the frequency axis
    """

    dx = _axis_x[1] - _axis_x[0]
    dy = _axis_y[1] - _axis_y[0]

    if _pad:
        pad_y = int(len(_axis_y) / 2)
        pad_x = int(len(_axis_x) / 2)
        _profile2D = np.pad(_profile2D, ((pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=0)

    m = _profile2D.shape[1]
    n = _profile2D.shape[0]
    psd = (dx * dy / (m * n)) * np.absolute(fft.fftshift(fft.fft2(_profile2D))) ** 2

    freqx = fft.fftshift(fft.fftfreq(m, dx))
    freqy = fft.fftshift(fft.fftfreq(n, dy))

    return psd, freqx, freqy


def uti_calc_averaged_psd(_profile2D, _axis_x, _axis_y, _pad=False):
    """
    This function receives a 2D height profile in [m] as numpy array (_profile2D) and two coordinate arrays (_axis_x and
    _axis_y) also in [m]. A 2D PSD is calculated by calling 'srw_uti_calc_2d_psd' and an azimuthal average is calculated.
    A 1D averaged PSD and it's accompanying axis is returned.

    :param _profile2D: height profile as a 2D numpy array in [m]
    :param _axis_x: 1D numpy array in [m]
    :param _axis_y: 1D numpy array in [m]
    :param _pad: (boolean) zero padding to increase the frequency sampling
    :return: 2D psd and its frequency axes
    """

    def _f_xy(_theta, _rho):
        x = _rho*np.cos(_theta)
        y = _rho*np.sin(_theta)
        return x, y

    psd_2d, fx, fy = uti_calc_2d_psd(_profile2D, _axis_x, _axis_y, _pad)

    xStart = fx[0]
    xFin = fx[-1]
    nx = fx.size

    yStart = fy[0]
    yFin = fy[-1]
    ny = fy.size

    # ********************** generating auxiliary vectors
    x_cen = 0.5*(xFin + xStart)
    y_cen = 0.5*(yFin + yStart)

    range_r = list((0, -1))
    range_theta = list((0, 2*np.pi))

    if xFin - x_cen > yFin - y_cen:
        range_r[1] = 1 * (yFin - y_cen)
    else:
        range_r[1] = 1 * (xFin - x_cen)

    range_theta[1] = 2*np.pi

    nr = int(nx*1/2)
    ntheta = int((range_theta[1] - range_theta[0])*360 * 1/2/np.pi)

    X = np.linspace(xStart, xFin, nx)
    Y = np.linspace(yStart, yFin, ny)

    R = np.linspace(range_r[0], range_r[1], nr)
    THETA = np.linspace(range_theta[0], range_theta[1], ntheta)

    psd_avg = np.zeros([nr])
    azimuthal_value = np.zeros([ntheta])

    # ********************** summation
    f = interp2d(X, Y, psd_2d)

    m = 0
    for rho in R:
        n = 0
        summation = 0
        for angle in THETA:
            x, y = _f_xy(angle, rho)
            azimuthal_value[n] = f(x, y)
            summation += 1
            n += 1

        psd_avg[m] = np.sum(azimuthal_value)/summation
        m = m+1

    return psd_avg, R


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Potpourri
# ----------------------------------------------------------------------------------------------------------------------

def calc_surf_rms(_arr):
    """

    :param _arr:
    :return:
    """
    x2 = np.multiply(_arr, _arr)
    SumX2 = np.nansum(x2)
    return np.sqrt((SumX2/np.count_nonzero(~np.isnan(x2))))


def uti_conj_sym_matrix(_mtx):
    """
    Apply conjugate symmetry to a 2D array (to be used with PSD calculations)
    :param _mtx: 2D numpy array
    :return:
    """
    n = _mtx.shape[0]
    m = _mtx.shape[1]
    _mtx[0, 0] = 0
    _mtx[0, int(m / 2)] = 0
    _mtx[int(n / 2), int(m / 2)] = 0
    _mtx[int(n / 2), 0] = 0
    _mtx[1::, 1:int(m / 2) + 1] = np.rot90(_mtx[1::, int(m / 2)::], 2)
    _mtx[0, 1:int(m / 2) + 1] = np.flipud(_mtx[0, int(m / 2)::])
    _mtx[int(n / 2)::, 0] = np.flipud(_mtx[1:int(n / 2) + 1, 0])
    _mtx[int(n / 2)::, int(m / 2)] = np.flipud(_mtx[1:int(n / 2) + 1, int(m / 2)])

    return _mtx


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