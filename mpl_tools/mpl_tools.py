#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy
import scipy, scipy.stats

def add_colorbar( colormapable, **kwargs ):
    rasterized = kwargs.pop( 'rasterized', True )

    ax = plt.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.gcf().colorbar( colormapable, cax=cax, **kwargs )
    if rasterized:
        cbar.solids.set_rasterized( True )
    plt.sca( ax )
    return cbar
##end add_colorbar

def chi2_nsigma( nsigma, ndf ):
    return scipy.stats.chi2.ppf( scipy.stats.chi2.cdf( nsigma**2, 1), ndf )
##end def get_chi2_nsigma

def chi2_v( sigmas, zmin=0.0 ):
    return [ chi2_nsigma( s, 2 )+zmin for s in sigmas ]
##end def chi2_v

def pop_existing( d, *args ):
    return [ d.pop(x, None) for x in args ]
##end def pop_existing

def savefig( name, *args, **kwargs ):
    """Save fig and print output filename"""
    if not name: return
    suffix, = pop_existing( kwargs, 'suffix' )
    if suffix!=None:
        from os.path import splitext
        basename, ext = splitext( name )
        name = basename+suffix+ext

    plt.savefig( name, *args, **kwargs )
    print( 'Save figure', name )
##end def savefig

def legend_set_alignment( legend, alignment='right', figure=None):
    """Set legend text alignment (to right)"""
    assert alignment=='right'
    figure = figure or plt.gcf()
    renderer = figure.canvas.get_renderer()
    shift = max([t.get_window_extent( renderer ).width for t in legend.get_texts()])
    for t in legend.get_texts():
        t.set_ha('right') # ha is alias for horizontalalignment
        t.set_position((shift,0))

def add_to_labeled( o, l ):
    ocurrent, lcurrent = ax.get_legend_handles_labels()
    ocurrent.append( o )
    lcurrent.append( l )

def legend_ext( before=[[],[]], after=[[],[]], ax=None, **kwargs ):
    alpha, = pop_existing( kwargs, 'alpha' )
    ax = ax or plt.gca()

    obefore, lbefore = before
    oafter, lafter = after
    ocurrent, lcurrent = ax.get_legend_handles_labels()
    leg = ax.legend( obefore+ocurrent+oafter, lbefore+lcurrent+lafter, **kwargs )
    if alpha!=None:
        leg.get_frame().set_alpha( alpha )
    return leg

def set_title( t ):
    """Set window title"""
    from pylab import canvas
    canvas.set_window_title( t )
##end if

def plot_hist( lims, height, *args, **kwargs ):
    """Plot histogram with lines. Like bar(), but without lines between bars."""
    zero_value = kwargs.pop( 'zero_value', 0.0 )
    y = numpy.empty( len(height)*2+2 )
    y[0], y[-1]=zero_value, zero_value
    y[1:-1] = numpy.vstack( ( height, height ) ).ravel( order='F' )
    x = numpy.vstack( ( lims, lims ) ).ravel( order='F' )

    return plt.plot( x, y, *args, **kwargs )
##end def plot_hist

def fill_between_hists( lims, hlower, hupper, *args, **kwargs ):
    """Plot histogram with lines. Like bar(), but without lines between bars."""
    label, = pop_existing( kwargs, 'label' )
    from numpy import ravel, empty, vstack
    if type(hlower)==float:
        yl = hlower
    else:
        yl = vstack( ( hlower, hlower ) ).ravel( order='F' )
    if type(hupper)==float:
        yu = hupper
    else:
        yu = vstack( ( hupper, hupper ) ).ravel( order='F' )
    x = vstack( ( lims[:-1], lims[1:] ) ).ravel( order='F' )

    if label:
        p = plt.Rectangle((0, 0), 0, 0, label=label, **kwargs)
        ax = plt.gca()
        ax.add_patch(p)

    #print( '  args', args, kwargs )
    #print( '  x', x )
    #print( '  lower', yl )
    #print( '  upper', yu )
    return plt.fill_between( x, yl, yu, *args, **kwargs )
##end def plot_hist

def fill_between_with_label(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else plt.gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = plt.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p
##end def

def plot_histbar( lims, height, *args, **kwargs ):
    fixedwidth, baropts = [ kwargs.pop(x) if x in kwargs else None\
                            for x in ['fixedwidth', 'baropts'] ]
    left  = lims[:-1]
    width = None
    if fixedwidth: width = lims[1]-lims[0]
    else: width = lims[1:] - left

    baropts = baropts or {}
    if not 'linewidth' in baropts: baropts['linewidth']=0

    from pylab import bar
    return plot_hist( lims, height, *args, **kwargs ), bar( left, height, width, **baropts )
##end def plot_histbar

def errorbar_array( buf, errs, *args, **kwargs ):
    weights, xoffset = [ kwargs.pop(x) if x in kwargs else None for x in ['weights', 'xoffset'] ]
    x = numpy.linspace( 0.5, len(buf)-0.5, len(buf) )
    if weights!=None:
        buf   = buf*weights
        errs *= weights
    ##end if weights

    xerrs = 0.5
    if xoffset:
        x+=xoffset
        xerrs = numpy.repeat( [ [0.5+xoffset], [0.5-xoffset] ], len(x), axis=1 )
    ##end if

    from matplotlib import pyplot as plt
    if not 'fmt' in kwargs: kwargs['fmt']=None
    return plt.errorbar( x, buf, errs, xerrs, *args, **kwargs )
##end def errorbar_array

def errorbar_ratio( a1, e1sq, a2, e2sq, *args, **kwargs ):
    w1, w2 = pop_existing( kwargs, 'weight1', 'weight2' )
    e1sq = e1sq if e1sq!=None else a1
    e2sq = e2sq if e2sq!=None else a2
    n = -3
    if w1!=None:
        a1   = a1*w1
        e1sq = e1sq*w1*w1
    ##end if
    if w2!=None:
        a2   = a2*w2
        e2sq = e2sq*w2*w2
    ##end if
    ratio = a1/a2
    d1sq = e1sq/a1**2
    d2sq = e2sq/a2**2
    err = ratio * ( d1sq+d2sq )**0.5

    # print( 'Ratio, err', ratio[n:], err[n:] )
    return errorbar_array( ratio, err, *args, **kwargs )
##end def errorbar_ratio

def plot_fcn( f, x, *args, **kwargs ):
    """Draw a function values for x"""
    from numpy import frompyfunc
    fun = frompyfunc( f, 1, 1 )

    return plt.plot( x, fun(x), *args, **kwargs )
##end def drawFun

def plot_table( text, loc=1, *args, **kwargs ):
    if type(text)==list:
        sep = kwargs.pop( 'separator', '\n')
        text = separator.join( text )

    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    # if bbox is None: bbox = dict(facecolor='white', alpha=1)
    if type(loc)==str:
        loc = {
            'upper right'  :    1
          , 'upper left'   :    2
          , 'lower left'   :    3
          , 'lower right'  :    4
          , 'right'        :    5
          , 'center left'  :    6
          , 'center right' :    7
          , 'lower center' :    8
          , 'upper center' :    9
          , 'center'       :    10
        }[loc]
    ##end if
    prop, = pop_existing( kwargs, 'prop' )
    prop = prop or {}
    at = AnchoredText( text, loc, *args, prop=prop, **kwargs )

    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.add_artist( at )
    return at
##end def plot_stats

def uses_latex():
    import matplotlib
    return matplotlib.rcParams[u'text.usetex']

