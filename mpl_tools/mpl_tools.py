#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy
import scipy, scipy.stats

def add_colorbar( colormapable, **kwargs ):
    rasterized = kwargs.pop( 'rasterized', True )
    minorticks = kwargs.pop( 'minorticks', False )
    minorticks_values = kwargs.pop( 'minorticks_values', None )

    ax = plt.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.gcf().colorbar( colormapable, cax=cax, **kwargs )

    if minorticks:
        if type(minorticks) is str:
            if minorticks=='linear':
                pass
            elif minorticks=='log':
                minorticks_values = colormapable.norm( minorticks_values )

            l1, l2 = cax.get_ylim()
            minorticks_values = minorticks_values[ (minorticks_values>=l1)*(minorticks_values<=l2) ]
            cax.yaxis.set_ticks(minorticks_values, minor=True)
        else:
            cax.minorticks_on()

    if rasterized:
        cbar.solids.set_rasterized( True )
    plt.sca( ax )
    return cbar
##end add_colorbar

def chi2_nsigma( nsigma, ndf ):
    return scipy.stats.chi2.ppf( scipy.stats.chi2.cdf( nsigma**2, 1), ndf )
##end def get_chi2_nsigma

def chi2_prob( levels, ndf ):
    return scipy.stats.chi2.ppf( levels, ndf )

def chi2_v( sigmas=None, probs=None, zmin=0.0 ):
    assert probs or sigmas
    if probs:
       return [ chi2_prob( p, 2 )+zmin for p in probs ]
    return [ chi2_nsigma( s, 2 )+zmin for s in sigmas ]
##end def chi2_v

def pop_existing( d, *args ):
    return [ d.pop(x, None) for x in args ]
##end def pop_existing

def savefig( name, *args, **kwargs ):
    """Save fig and print output filename"""
    if not name: return
    if type(name)==list:
        for n in name:
            savefig( n, *args, **kwargs.copy() )
        return

    suffix = kwargs.pop( 'suffix', None )
    addext = kwargs.pop( 'addext', [] )
    if suffix:
        from os.path import splitext
        basename, ext = splitext( name )
        name = basename+suffix+ext

    plt.savefig( name, *args, **kwargs )
    print( 'Save figure', name )

    if addext:
        if not type( addext )==list:
            addext = [ addext ]
        from os import path
        basename, extname = path.splitext( name )
        for ext in addext:
            name = '%s.%s'%( basename, ext )
            print( 'Save figure', name )
            plt.savefig( name, *args, **kwargs )
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
    ax = ax or plt.gca()

    if type(before[0]) is list:
        obefore, lbefore = before
    else:
        obefore = before
        lbefore = [ p.get_label() for p in before ]

    if type(after[0]) is list:
        oafter, lafter = after
    else:
        oafter = after
        lafter = [ p.get_label() for p in after ]

    ocurrent, lcurrent = ax.get_legend_handles_labels()

    leg = ax.legend( obefore+ocurrent+oafter, lbefore+lcurrent+lafter, **kwargs )
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

    if kwargs.pop( 'noedge', False ):
        x = x[1:-1]
        y = y[1:-1]

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

def plot_table( text, *args, **kwargs ):
    patchopts = kwargs.pop( 'patchopts', None )
    sep     = kwargs.pop( 'separator', u'\n')
    if type(text)==list:
        linefmt = kwargs.pop( 'linefmt', u'{}')
        if type( linefmt ) is str or type(linefmt) is unicode:
            fmt = linefmt
            linefmt = lambda *a: fmt.format( *a )
        lines = []
        lst = [linefmt(*line) if isinstance(line, (list,tuple)) else linefmt(line) for line in text]
        text = sep.join( lst )

    header, footer = kwargs.pop( 'header', None ), kwargs.pop( 'footer', None )
    if header: text = header+sep+text
    if footer: text = text+sep+footer

    if kwargs.pop( 'dump', False ):
        print( 'Table text:\n', text, sep='' )

    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    # if bbox is None: bbox = dict(facecolor='white', alpha=1)
    outside = kwargs.pop( 'outside', None )
    loc = kwargs.pop( 'loc', None )
    if outside:
        loc='upper left'
        kwargs.setdefault( 'borderpad', 0.0 )
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
    if not loc:
        loc = 1
    prop, = pop_existing( kwargs, 'prop' )
    prop = prop or {}
    ax = plt.gca()
    if outside:
        kwargs[ 'bbox_to_anchor' ]=outside
        kwargs[ 'bbox_transform' ]=ax.transAxes

    at = AnchoredText( text, loc, *args, prop=prop, **kwargs )
    if patchopts:
        at.patch.set( **patchopts )

    ax.add_artist( at )
    return at
##end def plot_stats

def uses_latex():
    import matplotlib
    return matplotlib.rcParams[u'text.usetex']

def append_levels( ax, ay, chi2levels, show_min=False, yax_min=0.0, **kwargs):
    axes       = kwargs.pop( 'axes', plt.gca() )
    text_sigma = kwargs.pop( 'text_sigma', True )
    textoffset = kwargs.pop( 'offset', 1 )
    flip = kwargs.pop( 'flip', False )
    topts = kwargs.pop( 'textopts', {} )
    interpolation = kwargs.pop( 'interpolation', 'linear' )
    imin = numpy.argmin( ay )
    xmin, ymin = ax[imin], ay[imin]

    from scipy.interpolate import interp1d
    from scipy.optimize import brentq, brent
    fcn = interp1d( ax, ay, kind=interpolation)

    if xmin==ax[0] or xmin==ax[-1]:
        brack = ( ax[0], ax[-1] )
    else:
        brack = ( ax[0], xmin, ax[-1] )
    print( 'bracket: ', brack )
    xmin = brent( fcn, brack=brack )
    ymin = fcn(xmin)
    if show_min:
        axes.vlines( [ xmin ], yax_min, ymin, **kwargs )
    levels = []
    for sign in chi2levels:
        level = ymin + sign**2
        try:
            x1 = brentq( lambda x: fcn(x)-level, ax[0], xmin )
        except:
            x1=0.0
        try:
            x2 = brentq( lambda x: fcn(x)-level, xmin, ax[-1] )
        except:
            print( 'Skip level for', sign )
            continue
        #print( 'Level', sign, level, x1, x2 )

        levels.append( [ x1, x2 ] )

        if flip:
            axes.hlines( [ x1, x2 ], yax_min, level, **kwargs )
            axes.vlines( level, x1, x2, **kwargs )
            if text_sigma:
                axes.text( level+textoffset, xmin, '%i$\sigma$'%sign, va='center', **topts )
        else:
            axes.vlines( [ x1, x2 ], yax_min, level, **kwargs )
            axes.hlines( level, x1, x2, **kwargs )
            if text_sigma:
                axes.text( (x1+x2)*0.5, level+textoffset, '%i$\sigma$'%sign, ha='center', va='bottom', **topts  )

    return levels

def hide_last_tick( axis, n=1 ):
    xticks = axis.get_major_ticks()
    if type(n) is int:
        n = slice( -n, None )
    for tick in xticks[n]:
        tick.label1.set_visible(False)

def modify_merged_axes( upper, lower, nticks=1 ):
    upper.tick_params( axis='x', which='both', bottom='off', labelbottom='off')
    upper.set_xlabel( '' )
    if nticks:
        hide_last_tick( lower.yaxis, n=nticks )

def indicate_outliers( ax, x=None, y=None, **kwargs ):
    kwargs.setdefault( 'color', 'red' )
    kwargs.setdefault( 'markersize', 8)
    if not x is None:
        if (x<ax.get_xlim()[0]).any():
            ax.plot( [0.01], [0.9], '<', transform=ax.transAxes, **kwargs )
        if (x>ax.get_xlim()[1]).any():
            ax.plot( [0.99], [0.9], '>', transform=ax.transAxes, **kwargs )

    if not y is None:
        if (y<ax.get_ylim()[0]).any():
            ax.plot( [0.9], [0.01], 'v', transform=ax.transAxes, **kwargs )
        if (y>ax.get_ylim()[1]).any():
            ax.plot( [0.9], [0.99], '^', transform=ax.transAxes, **kwargs )

def rebin( array, edges_from, edges_to, **kwargs ):
    """Rebin numpy array from one set of bin edges to another compatible set"""
    round = kwargs.pop( 'round', None )
    if not round is None:
        edges_from = numpy.round( edges_from, round )
        edges_to   = numpy.round( edges_to, round )

    mask    = numpy.in1d( edges_from, edges_to )
    indices = numpy.nonzero(mask)[0]
    if indices.size!=edges_to.size:
        print('Array edges are inconsistent:')
        print( 'from (%i): '%(edges_from.size), '\n', edges_from )
        print()
        print( 'to (%i): '%(edges_to.size),   '\n', edges_to )
        print()
        print( 'match indices (%i): '%(indices.size),    '\n', indices )
        print()
        print( 'matched elements (%i): '%(indices.size),    '\n', edges_from[mask] )
        print()

        assert False

    newarray = numpy.zeros( shape=edges_to.size-1, dtype=array.dtype )
    for j, (i1, i2) in enumerate(zip( indices[:-1], indices[1:] )):
        newarray[j] = array[i1:i2].sum()

    tolerance = kwargs.pop( 'tolerance', None )
    if not tolerance is None:
        s1, s2 = array.sum(), newarray.sum()
        diff = s1-s2
        if diff>tolerance:
            print( 'Error. Deviation above tolerance (%g):'%tolerance, s1, s2, diff )
            print( 'from (%i): '%(edges_from.size), '\n', edges_from )
            print( 'to (%i): '%(edges_to.size),   '\n', edges_to )

            raise Exception( 'deviation above tolerance' )

    return newarray
