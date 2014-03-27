#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy
import scipy, scipy.stats

def chi2_nsigma( nsigma, ndf ):
    return scipy.stats.chi2.ppf( scipy.stats.chi2.cdf( nsigma**2, 1), ndf )
##end def get_chi2_nsigma

def chi2_v( sigmas, zmin=0.0 ):
    return [ chi2_nsigma( s, 2 )+zmin for s in sigmas ]
##end def chi2_v

def pop_existing( d, *args ):
    return [ d.pop(x) if x in d else None for x in args ]
##end def pop_existing

def savefig( name, *args, **kwargs ):
    """Save fig and print output filename"""
    if not name: return
    from pylab import savefig
    savefig( name, *args, **kwargs )
    print( 'Save figure', name )
##end def savefig

def set_title( t ):
    """Set window title"""
    from pylab import canvas
    canvas.set_window_title( t )
##end if

def plot_hist( lims, height, *args, **kwargs ):
    """Plot histogram with lines. Like bar(), but without lines between bars."""
    y = numpy.empty( len(height)*2+2 )
    y[1:-1] = numpy.vstack( ( height, height ) ).ravel( order='F' )
    y[0]=0.0
    y[-1]=0.0
    x = numpy.vstack( ( lims, lims ) ).ravel( order='F' )

    from pylab import plot
    return plot( x, y, *args, **kwargs )
##end def plot_hist

def fill_between_hists( lims, hlower, hupper, *args, **kwargs ):
    """Plot histogram with lines. Like bar(), but without lines between bars."""
    from numpy import ravel, empty, vstack
    yl = vstack( ( hlower, hlower ) ).ravel( order='F' )
    yu = vstack( ( hupper, hupper ) ).ravel( order='F' )
    x = vstack( ( lims[:-1], lims[1:] ) ).ravel( order='F' )

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

def drawFun( f, x, *args, **kwargs ):
    """Draw a function values for x"""
    from numpy import frompyfunc
    fun = frompyfunc( f, 1, 1 )

    from pylab import plot
    return plot( x, fun(x), *args, **kwargs )
##end def drawFun

