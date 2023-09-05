#!/usr/bin/python
# coding: utf-8
###################################################################################
# barc4RefractiveOptics - example 2 - data from metrology and Zernike fit of the profile
# Authors/Contributors: Rafael Celestre
# Rafael.Celestre@esrf.eu
# creation: 12.10.2019
# last update: 12.10.2020
###################################################################################

import barc4ro.barc4ro as b4RO
from numpy import reshape, asarray

from oasys_srw.srwlib import *
from oasys_srw.uti_plot import *

def package_dirname(package):
    if isinstance(package, str): package = __import__(package, fromlist=[""])
    filename = package.__file__
    dirname = os.path.dirname(filename)
    return dirname


if __name__ == '__main__':
    print()
    print('barc4RefractiveOptics example #2:')
    print('Gaussian beam propagation through a 7-element Be CRL followed by a metrology profile from a real Be lens')
    print()
    save = False    # enables saving the intensity and phase
    plots = True    # displays intensity and phase

    strDataFolderName = "./"
    strIntPropOutFileName = "Intensity.dat"
    strPhPropOutFileName = "Phase.dat"

    #############################################################################
    #############################################################################
    # Beamline assembly
    beamE = 8000    # beam energy in eV

    R = 50 * 1E-6           # CRL radius at the parabola appex
    nCRL = 7                # number of lenses
    CRLAph = 440 * 1E-6     # CRL horizontal aperture
    CRLApv = 440 * 1E-6     # CRL vertical aperture
    delta = 5.3180778907372E-06
    attenuation_length = 5.276311E-3
    pinhole = 400 * 1E-6
    wt = 20.0 * 1E-6        # CRL wall thickness [um]
    shp = 1                 # 1- parabolic
    foc_plane = 3           # plane of focusing: 1- horizontal, 2- vertical, 3- both
    ContainerThickness = 2e-3   # space between the stacked lenses

    # ideal CRL
    CRL = b4RO.srwl_opt_setup_CRL(_foc_plane=foc_plane, _delta=delta, _atten_len=attenuation_length, _shape=shp,
                                 _apert_h=CRLAph, _apert_v=CRLApv, _r_min=R, _n=nCRL, _wall_thick=wt, _xc=0, _yc=0,
                                 _e_start=0, _e_fin=0, _nx=3001, _ny=3001, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0,
                                 _offst_ffs_x=0, _offst_ffs_y=0, _tilt_ffs_x=0, _tilt_ffs_y=0, _ang_rot_ez_ffs=0, _wt_offst_ffs=0,
                                 _offst_bfs_x=0, _offst_bfs_y=0, _tilt_bfs_x=0, _tilt_bfs_y=0, _ang_rot_ez_bfs=0, _wt_offst_bfs=0,
                                 isdgr=True)

    # Loading the metrology file and generating a transmission element

    heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(package_dirname("barc4ro.examples"), "metrology_data", "Be_50um_A440um_XSVT_metrology.dat"))
    oeCRL_Error = b4RO.srwl_opt_setup_CRL_metrology(_height_prof_data=heightProfData, _mesh=HPDmesh, _delta=delta,
                                                    _atten_len=attenuation_length, _wall_thick=0.0, _amp_coef=-7, _xc=0,
                                                    _yc=0, _ang_rot_ex=0, _ang_rot_ey=0, _ang_rot_ez=0, _fx=1e23,
                                                    _fy=1e23, isdgr=False)

    # The profile can be fit with a set of Zernike polynomials. Here we usw until the 37th pol. in order to obtain the
    # 3rd order spherical aberration
    N = 37
    heightProfData = reshape(asarray(heightProfData),(HPDmesh.ny,HPDmesh.nx))

    Zcoeffs, fit, residues = b4RO.fit_zernike_circ(heightProfData, nmodes=N, startmode=1, rec_zern=False)
    print('Zernike coefficients (um): \n' + str(Zcoeffs * 1e6))

    # pinhole
    oeApCRL = SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=pinhole)

    # drift space between lenses
    oeImagePosition = SRWLOptD(0.6791655113)

    # ============= Wavefront Propagation Parameters =======================#
    #                [ 0] [1] [2]  [3]  [4]  [5]  [6]  [7]   [8]  [9] [10] [11]
    ppCRL		    =[ 0,  0, 1.,   1,   0,  1., 10.,  1.,  10.,   0,   0,   0]
    ppCRL_Error	    =[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    ppAp		    =[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    ppImage         =[ 0,  0, 1.,   1,   0, 0.6,1.67, 0.6, 1.67,   0,   0,   0]
    ppFinal         =[ 0,  0, 1.,   1,   0, 0.5,  1., 0.5,   1.,   0,   0,   0]

    optBL = SRWLOptC([   CRL, oeCRL_Error, oeApCRL, oeImagePosition],
                     [ ppCRL, ppCRL_Error,    ppAp,         ppImage, ppFinal])

    """
    [ 3]: Type of Free-Space Propagator:
           0- Standard Fresnel
           1- Fresnel with analytical treatment of the quadratic phase terms
           2- Similar to 1, yet with different processing near a waist
           3- For propagation from a waist over a ~large distance
           4- For propagation over some distance to a waist
    [ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
    [ 6]: Horizontal Resolution modification factor at Resizing
    [ 7]: Vertical Range modification factor at Resizing
    [ 8]: Vertical Resolution modification factor at Resizing
    """

    #############################################################################
    #############################################################################
    # Photon source

    z = 60          # first optical element distance
    wfr_resolution = (256, 256)     # nx, ny
    screen_range = (-0.5E-3, 0.5E-3, -0.5E-3, 0.5E-3)       # x_Start, x_Fin, y_Start, y_Fin
    sampling_factor = 0.0       # sampling factor for adjusting nx, ny (effective if > 0)

    GsnBm = SRWLGsnBm()
    GsnBm.x = 0
    GsnBm.y = 0
    GsnBm.z = 0
    GsnBm.xp = 0
    GsnBm.yp = 0
    GsnBm.avgPhotEn = beamE
    GsnBm.pulseEn = 0.001
    GsnBm.repRate = 1
    GsnBm.polar = 1
    GsnBm.sigX = 0.5e-07 / 2.35
    GsnBm.sigY = GsnBm.sigX
    GsnBm.sigT = 10e-15
    GsnBm.mx = 0
    GsnBm.my = 0
    # Monochromatic wavefront
    wfr = SRWLWfr()
    wfr.allocate(1, wfr_resolution[0], wfr_resolution[1])  # Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.zStart = z
    wfr.mesh.eStart = beamE
    wfr.mesh.xStart = screen_range[0]
    wfr.mesh.xFin = screen_range[1]
    wfr.mesh.yStart = screen_range[2]
    wfr.mesh.yFin = screen_range[3]

    wfr.partBeam.partStatMom1.x = GsnBm.x
    wfr.partBeam.partStatMom1.y = GsnBm.y
    wfr.partBeam.partStatMom1.z = GsnBm.z
    wfr.partBeam.partStatMom1.xp = GsnBm.xp
    wfr.partBeam.partStatMom1.yp = GsnBm.yp

    arPrecPar = [sampling_factor]

    #############################################################################
    #############################################################################
    # Wavefront generation and beam propagation

    # ********************************Calculating Initial Wavefront and extracting Intensity:
    print('- Performing Initial Electric Field calculation ... ')
    srwl.CalcElecFieldGaussian(wfr, GsnBm, arPrecPar)
    print('Initial wavefront:')
    print('Nx = %d, Ny = %d' % (wfr.mesh.nx, wfr.mesh.ny))
    print('dx = %.4f um, dy = %.4f um' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E6 / wfr.mesh.nx,
                                                 (wfr.mesh.yFin - wfr.mesh.yStart) * 1E6 / wfr.mesh.ny))
    print('range x = %.4f mm, range y = %.4f mm' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E3,
                                                           (wfr.mesh.yFin - wfr.mesh.yStart) * 1E3))
    print('Rx = %.4f, Ry = %.4f' % (wfr.Rx, wfr.Ry))

    # ********************************Electrical field propagation
    print('- Simulating Electric Field Wavefront Propagation ... ')
    wfrp = deepcopy(wfr)
    srwl.PropagElecField(wfrp, optBL)

    print('Propagated wavefront:')
    print('Nx = %d, Ny = %d' % (wfrp.mesh.nx, wfrp.mesh.ny))
    print('dx = %.4f um, dy = %.4f um' % ((wfrp.mesh.xFin - wfrp.mesh.xStart) * 1E6 / wfrp.mesh.nx,
                                                 (wfrp.mesh.yFin - wfrp.mesh.yStart) * 1E6 / wfrp.mesh.ny))
    print('range x = %.4f um, range y = %.4f um' % ((wfrp.mesh.xFin - wfrp.mesh.xStart) * 1E6,
                                                           (wfrp.mesh.yFin - wfrp.mesh.yStart) * 1E6))
    print('Rx = %.10f, Ry = %.10f' % (wfrp.Rx, wfrp.Ry))

    if save is True or plots is True:
        arI1 = array('f', [0] * wfrp.mesh.nx * wfrp.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(arI1, wfrp, 6, 0, 3, wfrp.mesh.eStart, 0, 0)
        arP1 = array('d', [0] * wfrp.mesh.nx * wfrp.mesh.ny)  # "flat" array to take 2D phase data (note it should be 'd')
        srwl.CalcIntFromElecField(arP1, wfrp, 0, 4, 3, wfrp.mesh.eStart, 0, 0)

    if save:
        srwl_uti_save_intens_ascii(arI1, wfrp.mesh, os.path.join(os.getcwd(), strDataFolderName, strIntPropOutFileName),0)
        srwl_uti_save_intens_ascii(arP1, wfrp.mesh, os.path.join(os.getcwd(), strDataFolderName, strPhPropOutFileName),0)
    print('>> single electron calculations: done')

    if plots is True:
        # ********************************Electrical field intensity and phase after propagation
        plotMesh1x = [1E6 * wfrp.mesh.xStart, 1E6 * wfrp.mesh.xFin, wfrp.mesh.nx]
        plotMesh1y = [1E6 * wfrp.mesh.yStart, 1E6 * wfrp.mesh.yFin, wfrp.mesh.ny]
        uti_plot2d(arI1, plotMesh1x, plotMesh1y,
                   ['Horizontal Position [um]', 'Vertical Position [um]', 'Intensity After Propagation'])
        uti_plot2d(arP1, plotMesh1x, plotMesh1y,
                   ['Horizontal Position [um]', 'Vertical Position [um]', 'Phase After Propagation'])

        uti_plot_show()

