#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import ROOT, r2numpy, numpy
from mpl_tools import *
from matplotlib import pyplot as plt
import numpy as np
import itertools

def plot_graph( g, *args, **kwargs ):
    """Plot TGraph"""
    from pylab import plot
    x, y = r2numpy.get_buffers_graph( g ).copy()
    return plot( x, y, *args, **kwargs )
##end def

def errorbar_graph( g, *args, **kwargs ):
    """Plot TGraphErrors"""
    from pylab import errorbar
    x, y = g.get_buffers().copy()
    ex, ey = g.get_err_buffers() #FIXME: where is it?
    from numpy import all
    ex = ex if all( ex!=0.0 ) else None
    ey = ey if all( ey!=0.0 ) else None
    return errorbar( x, y, ey, ex, *args, **kwargs )
##end def

def errorbar_graph_asymm( g, *args, **kwargs ):
    """Plot TGraphErrors"""
    from pylab import errorbar
    x, y = g.get_buffers().copy()
    exl, exh, eyl, eyh = g.get_err_buffers() #FIXME: where is it?
    from numpy import array
    ex = array( ( exl, exh ) )
    ey = array( ( eyl, eyh ) )
    from numpy import all
    ex = ex if all( ex!=0.0 ) else None
    ey = ey if all( ey!=0.0 ) else None
    return errorbar( x, y, ey, ex, *args, **kwargs )
##end def

def contour_graph2d( graph, shape, *args, **kwargs ):
    x, y, z = r2numpy.get_buffers_mat_graph2d( graph, shape ).copy()
    if x is None:
        return

    chi2levels, v = pop_existing( kwargs,  'chi2levels', 'v'  )
    if chi2levels:
        v = chi2_v( chi2levels, z.min() )
        return plt.contour( x, y, z, v, *args, **kwargs ), dict( x=x, y=y, z=z, v=v )

    return plt.contour( x, y, z, *args, **kwargs ), dict( x=x, y=y, z=z, v=None )
##end def contour_graph2d

def contour_hist2( hist, *args, **kwargs ):
    z = r2numpy.get_buffer_hist2( hist ).copy()
    X = r2numpy.get_bin_centers_axis( hist.GetXaxis() )
    Y = r2numpy.get_bin_centers_axis( hist.GetYaxis() )

    chi2levels, v, xyflip = pop_existing( kwargs,  'chi2levels', 'v', 'xyflip'  )
    if xyflip:
        z = z.T
        X, Y = Y, X

    x, y = np.meshgrid( X, Y )

    if chi2levels:
        v = chi2_v( chi2levels, z.min() )
        return plt.contour( x, y, z, v, *args, **kwargs ), dict( x=x, y=y, z=z, v=v )

    return plt.contour( x, y, z, *args, **kwargs ), dict( x=x, y=y, z=z, v=None )
##end def contour_graph2d

def contourf_hist2( hist, *args, **kwargs ):
    z = r2numpy.get_buffer_hist2( hist )
    X = r2numpy.get_bin_centers_axis( hist.GetXaxis() )
    Y = r2numpy.get_bin_centers_axis( hist.GetYaxis() )

    x, y = np.meshgrid( X, Y )

    chi2levels, v = pop_existing( kwargs,  'chi2levels', 'v'  )
    if chi2levels:
        v = chi2_v( chi2levels, z.min() )
        return plt.contourf( x, y, z, v, *args, **kwargs ), dict( x=x, y=y, z=z, v=v )

    return plt.contourf( x, y, z, *args, **kwargs ), dict( x=x, y=y, z=z, v=None )
##end def contour_graph2d

def contourf_graph2d( graph, shape, *args, **kwargs ):
    x, y, z =r2numpy.get_buffers_mat_graph2d( graph, shape ).copy()
    if x is None:
        return

    chi2levels, v = pop_existing( kwargs,  'chi2levels', 'v'  )
    if chi2levels:
        v = chi2_v( chi2levels, z.min() )
        return plt.contourf( x, y, z, v, *args, **kwargs ), dict( x=x, y=y, z=z, v=v )

    return plt.contourf( x, y, z, *args, **kwargs ), dict( x=x, y=y, z=z, v=None )
##en##end def contour_graph2d

def plot_surface_graph2d( graph, shape, *args, **kwargs ):
    x, y, z = r2numpy.get_buffers_mat_graph2d( graph, shape ).copy()
    if x is None:
        return

    colorbar, = pop_existing( kwargs, 'colorbar' )
    res = plt.gca().plot_surface( x, y, z, *args, **kwargs )
    cbar = add_colorbar( res ) if colorbar else None

    return res, dict( x=x, y=y, z=z, colorbar=cbar )
##end def contour_graph2d

def plot_trisurf_graph2d( graph, *args, **kwargs ):
    x, y, z = r2numpy.get_buffers_graph2d( graph ).copy()
    if x is None:
        return
    colorbar, = pop_existing( kwargs, 'colorbar' )
    res = plt.gca().plot_trisurf( x, y, z, *args, **kwargs )
    cbar = add_colorbar( res ) if colorbar else None

    return res, dict( x=x, y=y, z=z, colorbar=cbar )
##end def contour_graph2d

def bar_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.bar"""
    height = r2numpy.get_buffer_hist1( h ).copy()
    ax = h.GetXaxis()
    lims, fixed = r2numpy.get_bin_edges_axis( ax, type=True )
    width=None
    left  = lims[:-1]
    if fixed: width = ax.GetBinWidth( 1 )
    else: width = lims[1:] - left

    from pylab import bar
    return bar( left, height, width, *args, **kwargs )
##end plot_bar_hist1

def errorbar_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot"""
    noyerr, mask, = [ kwargs.pop(x) if x in kwargs else None for x in ['noyerr', 'mask'] ]
    centers = r2numpy.get_bin_centers_axis( h.GetXaxis())
    hwidths = r2numpy.get_bin_widths_axis( h.GetXaxis())*0.5
    height=r2numpy.get_buffer_hist( h ).copy()
    if not mask is None: height = numpy.ma.array( height, mask=mask )

    yerr = None
    if not noyerr:
        yerr2 = r2numpy.get_err_buffer_hist1( h )
        if yerr2 is None: yerr2 = height
        yerr = yerr2**0.5
    ##end if

    if not 'fmt' in kwargs: kwargs['fmt']=None
    from matplotlib.pyplot import errorbar
    from numpy import sqrt
    return centers, height, errorbar( centers, height, yerr, hwidths, *args, **kwargs )
##end plot_bar_hist1

def plot_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot"""
    lims=r2numpy.get_bin_edges_axis(h.GetXaxis())
    height=r2numpy.get_buffer_hist1( h ).copy()
    return plot_hist( lims, height, *args, **kwargs )
##end plot_bar_hist1

def plot_stack_hist1( hists, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot"""
    zero_value, colors, labels, zorder = pop_existing( kwargs, 'zero_value', 'colors', 'labels', 'zorder' )
    if zero_value is None:
        zero_value = 0.0
    if not labels:
        labels = []
    if type(zorder)!=list:
        zorder=[ zorder ]
    colors = colors or 'rgbcmy'

    previous = zero_value
    ret = []
    for h, color, label, z in itertools.izip( hists, itertools.cycle( colors ), itertools.cycle( labels ), itertools.cycle( zorder ) ):
        lims=r2numpy.get_bin_edges_axis( h.GetXaxis() )
        height=previous + r2numpy.get_buffer_hist1( h )
        res = fill_between_hists( lims, previous, height, *args, color=color, label=label, zorder=z, **kwargs )
        ret.append( res )
        previous = height
    return ret
##end plot_bar_hist1

def plot_centers_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot"""
    mask, = pop_existing( kwargs, 'mask' )
    lims=r2numpy.get_bin_edges_axis(h.GetXaxis())
    x = r2numpy.get_bin_centers_axis( h.GetXaxis() )
    y = r2numpy.get_buffer_hist1( h ).copy()
    if not mask is None: y = numpy.ma.array( y, mask=y==mask )
    return x, y, plt.plot( x, y, *args, **kwargs )
##end plot_bar_hist1

def plot_bar_hist1( h, *args, **kwargs ):
    """Plot 1-dimensinal histogram using pyplot.plot and pyplot.bar
       baroptions are passed as baropts=dict(opt=value)
    """
    ax = h.GetXaxis()
    height = r2numpy.get_buffer_hist1( h ).copy()
    lims, fixed = r2numpy.get_bin_edges_axis(ax, type=True )

    from mpl_tools import plot_histbar
    return plot_histbar( lims, height, *args, fixedwidth=fixed, **kwargs )
##end plot_bar_hist1

def plot_date(h, t0, *args, **kwargs):
    from matplotlib import pyplot as plt
    from matplotlib.dates import date2num
    from datetime import date

    days=h.GetNbinsX()-2
    weeks=days/7
    rem=days%7
    times=[ int(t0 + ( i - 0.5 ) * 7 * 86400) for i in range(1, weeks+1) ]
    t=int(t0 + ( 7 * i - rem/2. ) * 86400)
    times.append( t )
    timestamps=[ date.fromtimestamp(ts) for ts in times ]
    dates=date2num(timestamps)
    data=[]
    a=h.GetArray()
    for i in range( 1, weeks+1 ):
        R=0
        for j in range( 1, 8 ):
            R+=a[i*j]
        ##end for j
        R/=7.
        data.append(R)
    ##end for i
    R=0.
    for i in range( 1, rem+1 ):
        R+=a[days+1-rem+i]
    ##end for i
    R/=rem
    data.append(R)
    return plt.plot_date( dates, data, *args, **kwargs )
##end def plot_date

def get_stats_hist( hist, opt='nemr', definitions={} ):
    names = {  'k' : 'Kurtosis'
             , 's' : 'Skewness'
             , 'i' : 'Integral'
             , 'o' : 'Overflow'
             , 'u' : 'Underflow'
             , 'r' : 'RMS'
             , 'm' : 'Mean'
             , 'M' : 'Mean (RMS)'
             , 'e' : 'Entries'
             , 'n' : 'Name' }

    methods = None
    if isinstance( hist, ROOT.TH1 ):
        methods = {
              'k' : [ hist.GetKurtosis ]
            , 'K' : [ hist.GetKurtosis, lambda: hist.GetKurtosis(11) ]
            , 's' : [ hist.GetSkewness ]
            , 'S' : [ hist.GetSkewness, lambda: hist.GetSkewness(11) ]
            , 'i' : [ hist.Integral ]
            , 'o' : [ lambda: hist.GetArray()[hist.GetNbinsX()+1] ]
            , 'u' : [ lambda: hist.GetArray()[0] ]
            , 'r' : [ hist.GetRMS ]
            , 'R' : [ hist.GetRMS, lambda: hist.GetRMS(11) ]
            , 'm' : [ hist.GetMean ]
            , 'M' : [ hist.GetMean, lambda: hist.GetMean(11) ]
            , 'e' : [ hist.GetEntries ]
            , 'n' : [ hist.GetName ]
        }
    elif isinstance( hist, numpy.ndarray ):
        from scipy import stats
        methods = {
              'i' : [ lambda: len(hist) ]
            , 'k' : [ lambda: stats.kurtosis( hist ) ]
            , 's' : [ lambda: stats.skew( hist ) ]
            , 'r' : [ hist.std ]
            , 'm' : [ hist.mean ]
            , 'M' : [ lambda: hist.std()/len(hist)**0.5 ]
            , 'e' : [ lambda: len( hist ) ]
        }
    else: assert False, 'Unknown type to build stats '+str(type(hist))

    stats={}
    for mode in opt:
        name, fcn = ( names.get( mode ) or names[mode.lower()] ), methods.get( mode )
        predef = definitions.get( name.lower() )
        assert ( fcn or predef ) and name
        stats[ name ] = [ predef ] if predef!=None else [ f() for f in fcn ]
    ##end for

    return stats
##end if

def get_stats( hist, opt='nemr', *args, **kwargs ):
    stats = get_stats_hist( hist, opt, *args, **kwargs )
    table_vals={}
    for key, v in stats.iteritems():
        if ( key != 'Name' ):
            if (len(v)==2): val='{0:.5f}'.format(round(v[0], 5))+r'\pm'+'{0:.8f}'.format(round(v[1], 8))
            else:
                if (v[0]==int(v[0])): val = '{0:.0f}'.format(v[0])
                else: val = '{0:.5f}'.format(round(v[0], 5))
            table_vals.update( { key : val } )
        ##end if
    ##end for key
    from matplotlib import rc
    rc('text', usetex=True)
    statlist=[ 'Entries', 'Mean', 'Mean (RMS)', 'RMS', 'Underflow', 'Overflow', 'Integral', 'Skewness', 'Kurtosis' ]
    table=r'\begin{tabular}{ |l r| }    \hline '
    if 'Name' in stats:
        table+=r'\multicolumn{2}{|c|}{'+str(stats['Name'][0])+r'} \\ \hline '
    for stat in statlist:
        if stat in table_vals:
            table+=stat+r' & '+table_vals[stat]+r'\\'
        ##end if
    ##end for
    table+=r'\hline \end{tabular}'
    return table
##end def get_stats

def plot_stats( self, opt='nemr', loc=1, definitions={}, *args, **kwargs ):
    s = get_stats( self, opt, definitions )
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
    at = AnchoredText( s, loc, *args, frameon=False, prop={}, **kwargs )
    at.patch.set_facecolor( 'white' )
    at.patch.set_color( 'white' )
    at.patch.set_fill( 'full' )
    at.patch.set_alpha( 1.0 )
    at.set_alpha( 1.0 )
    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.add_artist( at )
    return at
##end def plot_stats

def pcolor_mesh_hist2( h, *args, **kwargs ):
    """Plot TH2 using matplotlib.pcolormesh"""
    mask, colz, colorbar = pop_existing( kwargs, 'mask', 'colz', 'colorbar' )

    # get bin edges first
    x1 = r2numpy.get_bin_edges_axis(h.GetXaxis())
    y1 = r2numpy.get_bin_edges_axis(h.GetYaxis())

    # make a 2D mesh
    from numpy import meshgrid
    x, y = meshgrid( x1, y1 )
    # print( 'mesh x', x )
    # print( 'mesh y', y )

    # get data bufer w/o underflow/overflow bins
    buf = r2numpy.get_buffer_hist2( h,  mask=mask ).copy()

    # plot
    from pylab import pcolormesh
    res = pcolormesh( x, y, buf, *args, **kwargs )
    cbar = add_colorbar( res ) if colz or colorbar else None

    return res
##end def pcolor_hist2

def pcolor_hist2( h, *args, **kwargs ):
    """Plot TH2 using matplotlib.pcolorfast"""
    mask, colz, colorbar = pop_existing( kwargs, 'mask', 'colz', 'colorbar' )

    # get bin edges first
    xax = h.GetXaxis()
    yax = h.GetYaxis()
    if xax.GetXbins().GetSize()>0 or yax.GetXbins().GetSize()>0:
        print( 'Can not draw 2D a histogram with variable bin widths' )
        print( 'Use pcolormesh method or draweHist2Dmesh function instead' )
        return
    ##end if
    x = [ xax.GetXmin(), xax.GetXmax() ]
    y = [ yax.GetXmin(), yax.GetXmax() ]

    # get data bufer w/o underflow/overflow bins
    buf = r2numpy.get_buffer_hist2( h,  mask=mask ).copy()

    # plot
    from pylab import axes
    ax = axes()
    res = ax.pcolorfast( x, y, buf, *args, **kwargs )
    cbar = add_colorbar( res ) if colorbar or colz else None
    if cbar:
        return res,cbar

    return res
##end def pcolor_hist2

def imshow_hist2( h, *args, **kwargs ):
    """Plot TH2 using matplotlib.pcolorfast"""
    mask, colorbar = pop_existing( kwargs, 'mask', 'colorbar' )

    # get bin edges first
    xax = h.GetXaxis()
    yax = h.GetYaxis()
    if xax.GetXbins().GetSize()>0 or yax.GetXbins().GetSize()>0:
        print( 'Can not draw 2D a histogram with variable bin widths' )
        print( 'Use pcolormesh method or draweHist2Dmesh function instead' )
        return
    ##end if
    extent = [ xax.GetXmin(), xax.GetXmax(), yax.GetXmin(), yax.GetXmax()  ]

    # get data bufer w/o underflow/overflow bins
    buf = r2numpy.get_buffer_hist2( h,  mask=mask ).copy()

    res = plt.imshow( buf, *args, extent=extent, **kwargs )
    cbar = add_colorbar( res ) if colorbar or colz else None
    if cbar:
        return res,cbar

    return res
##end def pcolor_hist2

def imshow_matrix( self, *args, **kwargs ):
    """Plot TMatrixD using matplotlib.imshow"""
    mask, colorbar = \
        pop_existing( kwargs, 'mask', 'colorbar' )

    buf = r2numpy.get_buffer_matrix( self ).copy()
    if not mask is None:
        buf = np.ma.array( buf, mask = buf==mask )
    ##end if

    res = plt.imshow( buf, *args, **kwargs )
    cbar = add_colorbar( res ) if colorbar else None
    if cbar:
        return res, cbar

    return res
##end def pcolor_hist2

def pcolor_matrix( m, *args, **kwargs ):
    """Plot TMatrixD using matplotlib.pcolorfast"""
    mask, colz, colorbar, limits = \
        pop_existing( kwargs, 'mask', 'colz', 'colorbar', 'limits' )
    x, y = limits!=None and limits or ([ 0.0, m.GetNcols() ], [ 0.0, m.GetNrows() ])

    buf = r2numpy.get_buffer_matrix( m ).copy()
    if not mask is None:
        from numpy import ma
        buf = ma.array( buf, mask = buf==mask )
    ##end if

    # plot
    from pylab import axes
    ax = axes()
    res = ax.pcolorfast( x, y, buf, *args, **kwargs )
    cbar = add_colorbar( res ) if colz or colorbar else None
    if cbar:
        return res, cbar

    return res
##end def pcolor_hist2

def pcolor_mesh_matrix( h, xedges, yedges, *args, **kwargs ):
    """Plot TMatrix using matplotlib.pcolormesh"""
    # make a 2D mesh
    from numpy import meshgrid
    x, y = meshgrid( xedges, yedges )

    # get buffer
    buf = r2numpy.get_buffer_matrix( h ).copy()

    # plot
    from pylab import pcolormesh
    res = pcolormesh( x, y, buf, *args, **kwargs )
    return res
##end def pcolor_hist2

def plot_diag_matrix( m, *args, **kwargs ):
    """Plot TH2 diagoanl ad TH1"""
    assert m.GetNcols()==m.GetNrows(), 'Matrix is not square'
    limits, = [ kwargs.pop(x) if x in kwargs else None for x in ['limits'] ]

    buf = r2numpy.get_buffer_matrix( m ).copy()
    from numpy import diagonal
    bins = diagonal( buf )
    lims = None
    if limits:
        from numpy import linspace
        lims = linspace( limits[0], limits[1], m.GetNcols()+1 )
    else:
        from numpy import arange
        lims = arange( 0.0, m.GetNcols()+1, 1.0 )
    ##end if

    from mpl_tools import plot_histbar
    return plot_histbar( lims, bins, *args, **kwargs )
##end def pcolor_hist2

def pcolor_diag_hist2( h, *args, **kwargs ):
    """Plot TH2 diagoanl ad TH1"""
    assert h.GetNbinsX()==h.GetNbinsY(), 'Histogram is not square'
    buf = r2numpy.get_buffer_hist2( h ).copy()
    from numpy import diagonal
    bins = diagonal( buf )
    lims, fixedwidth = r2numpy.get_bin_edges_axis( h.GetXaxis(), type=True )

    from mpl_tools import plot_histbar
    return plot_histbar( lims, bins, *args, fixedwidth=fixedwidth, **kwargs )
##end def pcolor_hist2

def plot_f1( f, x=None, *args, **kwargs ):
    """Plot TF1
       if x is an array-like it's used as array of x
       if x is integer it is used as N of points in function range
       if x is float it is used as step
       if x is None, the TF1->Npx is used as number of points
    """
    import numpy
    tp = type(x)
    if x is None:
        x = numpy.linspace( f.GetXmin(), f.GetXmax(), f.GetNpx() )
    elif tp==int:
        x = numpy.linspace( f.GetXmin(), f.GetXmax(), x )
    elif tp==float:
        x = numpy.arange( f.GetXmin(), f.GetXmax(), x )
    ##end

    return drawFun( f.Eval, x, *args, **kwargs )
##end def plot_f1

def errorbar_array( self, *args, **kwargs ):
    from mpl_tools import errorbar_array
    buf = r2numpy.get_buffer_array( self ).copy()
    errs = None
    if args:
        errs = args[0]
        args = args[1:]
    ##end if args
    errs = ( buf if errs is None else errs )**0.5

    errorbar_array( buf, errs, *args, **kwargs )
##end def TArrayD

def get_fit_result( fun, opt='cev' ):
    res={}
    if ( opt.find('p') != -1 ):
        p={ 'Prob' : fun.GetProb() }
        res.update(p)
    ##end if
    if ( opt.find('c') != -1 ):
        c={ 'Chisq' : [ fun.GetChisquare(), fun.GetNDF() ] }
        res.update(c)
    ##end if
    if ( opt.find('e') != -1 ):
        p={ 'Pars' : [] }
        npars=fun.GetNpar()
        for i in range(npars):
            name=fun.GetParName(i)
            value=fun.GetParameter(i)
            error=fun.GetParError(i)
            x=[name, value, error]
            p['Pars'].append(x)
        ##end for i
        res.update(p)
    ##end if
    if ( opt.find('e') == -1 and opt.find('v') != -1 ):
        p={ 'Pars' : [] }
        npars=fun.GetNpar()
        for i in range(npars):
            name=fun.GetParName(i)
            value=fun.GetParameter(i)
            p['Pars'].append([name, value])
        ##end for i
        res.update(p)
    ##end if

    table_vals={}
    rows = [ r'$\chi^2$ / ndf ', 'Prob']
    for par in res['Pars']: rows.append(par[0])
    if 'Chisq' in res:
        name = r'$\chi^2$ / ndf '
        val = '{0:.3f}'.format( round( res['Chisq'][0], 3 ) ) + r' / ' + str( res['Chisq'][1] )
        table_vals.update( { name : val } )
    ##end if
    if 'Prob' in res:
        name = 'Prob'
        val = str( res['Prob'] )
        table_vals.update( { name : val } )
    ##end if
    if 'Pars' in res:
        for par in res['Pars']:
            name = par[0]
            val = '{0:.5f}'.format( round( par[1], 5 ) )
            if(len(par)==3): val += r'$\pm$' + '{0:.8f}'.format( round( par[2], 8 ) )
            table_vals.update( { name : val } )
        ##end for par
    ##end if
    from matplotlib import rc
    rc('text', usetex=True)
    table=r'\begin{tabular}{ |l r| }    \hline '
    for v in rows:
        if v in table_vals:
            table += v + r' & ' + table_vals[v] + r'\\'
        ##end if
        ##end for v
    table+=r'\hline \end{tabular}'
    return table
##end def get_fit_result

def import_title( o, what='' ):
    """Import object titles to canvas/axis labels"""
    if what=='': return
    from pylab import axes
    ax = axes()
    if 'x' in what: ax.set_xlabel( o.GetXaxis().GetTitle() )
    if 'y' in what: ax.set_ylabel( o.GetYaxis().GetTitle() )
    if 't' in what: ax.set_title( o.GetTitle() )
    if 'n' in what: ax.set_title( o.GetName() )
    if 'f' in what: ax.set_title( o.GetExpFormula() )
##end def import_title

def bind_functions():
    setattr( ROOT.TArray, 'errorbar', errorbar_array )

    setattr( ROOT.TF1, 'show_title', import_title )
    setattr( ROOT.TF1, 'get_fit_result', get_fit_result )
    setattr( ROOT.TF1, 'plot', plot_f1 )

    setattr( ROOT.TGraph, 'plot', plot_graph )
    setattr( ROOT.TGraph, 'show_title', import_title )

    setattr( ROOT.TGraphAsymmErrors, 'errorbar', errorbar_graph_asymm )
    setattr( ROOT.TGraphAsymmErrors, 'plot', plot_graph )

    setattr( ROOT.TGraphErrors, 'errorbar', errorbar_graph )
    setattr( ROOT.TGraphErrors, 'plot', plot_graph )

    setattr( ROOT.TGraph2D, 'plot_surface',  plot_surface_graph2d )
    setattr( ROOT.TGraph2D, 'plot_trisurf',  plot_trisurf_graph2d )
    setattr( ROOT.TGraph2D, 'contour',  contour_graph2d )
    setattr( ROOT.TGraph2D, 'contourf', contour_graph2d )

    setattr( ROOT.TH1, 'show_title', import_title )
    setattr( ROOT.TH1, 'bar', bar_hist1 )
    setattr( ROOT.TH1, 'draw', plot_bar_hist1 )
    setattr( ROOT.TH1, 'errorbar', errorbar_hist1 )
    setattr( ROOT.TH1, 'get_stats', get_stats )
    setattr( ROOT.TH1, 'plot', plot_hist1 )
    setattr( ROOT.TH1, 'plot_centers', plot_centers_hist1 )
    setattr( ROOT.TH1, 'plot_stats', plot_stats )
    setattr( ROOT.TH1, 'plot_date', plot_date )
    setattr( ROOT.TH1, 'plot_stack', lambda s, *a, **k: plot_stack_hist1( self, *a, **k ) )

    setattr( ROOT.TH2, 'draw', pcolor_hist2 )
    setattr( ROOT.TH2, 'plot_diag', pcolor_diag_hist2 )
    setattr( ROOT.TH2, 'plot_mesh', pcolor_mesh_hist2 )
    setattr( ROOT.TH2, 'pcolorfast', pcolor_hist2 )
    setattr( ROOT.TH2, 'pcolormesh', pcolor_mesh_hist2 )
    setattr( ROOT.TH2, 'imshow', imshow_hist2 )
    setattr( ROOT.TH2, 'contour', contour_hist2 )
    setattr( ROOT.TH2, 'contourf', contourf_hist2 )

    setattr( ROOT.TMatrixD, 'plot_diag', plot_diag_matrix )
    setattr( ROOT.TMatrixD, 'pcolorfast', pcolor_matrix )
    setattr( ROOT.TMatrixD, 'pcolormesh', pcolor_mesh_matrix )
    setattr( ROOT.TMatrixD, 'imshow', imshow_matrix )

    setattr( ROOT.TMatrixF, 'plot_diag', plot_diag_matrix )
    setattr( ROOT.TMatrixF, 'pcolorfast', pcolor_matrix )
    setattr( ROOT.TMatrixF, 'pcolormesh', pcolor_mesh_matrix )
    setattr( ROOT.TMatrixF, 'imshow', imshow_matrix )
##end def function
