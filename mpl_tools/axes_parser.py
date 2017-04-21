#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from argparse_utils import SingleSubparser
from argparse import ArgumentParser
# from defs import translate, Translate
from mpl_tools import hide_last_tick

def ApplyAxesOptions( opts, ax=None, fig=None ):
    if not opts:
        return None, None, None
    from matplotlib import pyplot as plt

    fig = fig or opts.fig
    ax  = ax or opts.axis
    if ax is None or type(ax) is int:
        if fig:
            if type(fig) is str:
                fig = plt.figure( fig )
            ax = fig.axes[ ax if type(ax) is int else 0 ]
        else:
            ax = plt.gca()
            fig = plt.gcf()
    ax.grid( opts.grid )
    if opts.title:
        ax.set_title( opts.title )
    if opts.x_lim:
        ax.set_xlim( opts.x_lim )
    if opts.y_lim:
        ax.set_ylim( opts.y_lim )
    if opts.z_lim:
        ax.set_zlim( opts.z_lim )
    if opts.xlog:
        ax.set_xscale( 'log' )
    if opts.ylog:
        ax.set_yscale( 'log' )

    if opts.legend:
        leg = ax.legend( loc=opts.legend_location, title=opts.legend_title,
                         framealpha=opts.legend_framealpha,
                         ncol=opts.legend_ncol )
    else:
        leg = None

    if opts.hide_last_ticks:
        hide_last_tick( ax.yaxis, opts.hide_last_ticks )

    return fig, ax, leg

def AxesSubparser( prefix_char='+', args_included=[], args_excluded=[], defaults={}, single=False ):
    fmt = dict( prefix=prefix_char, p=prefix_char )
    arguments = dict( [
           #( 'labels',          [ [ '{p}l'.format( **fmt ) ], dict( nargs='+', help='labels' ) ] ),
           ( 'title',             [ [ '{p}t'.format( **fmt ) ], dict( default='', help='title' ) ]  ),
           ( 'no-grid',           [ [ '{p}G'.format( **fmt ) ], dict( action='store_false', dest='grid', help='no grid' ) ] ),
           ( 'grid',              [ [ '{p}g'.format( **fmt ) ], dict( action='store_true', help='grid' ) ] ),
           ( 'xlog',              [ [], dict( action='store_true', help='grid' ) ] ),
           ( 'ylog',              [ [], dict( action='store_true', help='grid' ) ] ),
           ( 'legend',            [ [ '{p}l'.format( **fmt ) ], dict( action='store_true', help='legend' )] ),
           ( 'legend-title',      [ [ '{p}{p}lt'.format(**fmt) ], dict( help='legend title' )] ),
           ( 'legend-location',   [ [ '{p}L'.format( **fmt ) ], dict( default='upper right', help='legend location') ]  ),
           ( 'legend-framealpha', [ [ '{p}{p}lfa'.format(**fmt) ], dict( type=float, help='legend frame alpha') ]  ),
           ( 'legend-ncol',       [ [ '{p}{p}lnc'.format(**fmt) ], dict( type=int, help='legend columns') ]  ),
           ( 'x-lim',             [ [ '{p}x'.format( **fmt ) ], dict( type=float, nargs=2, help='x limits') ] ),
           ( 'y-lim',             [ [ '{p}y'.format( **fmt ) ], dict( type=float, nargs=2, help='y limits') ] ),
           ( 'z-lim',             [ [ '{p}z'.format( **fmt ) ], dict( type=float, nargs=2, help='z limits') ] ),
           ( 'axis',              [ [ '{p}a'.format( **fmt ) ], dict( type=int, help='axes number' ) ] ),
           ( 'fig',               [ [ '{p}f'.format( **fmt ) ], dict( type=int, help='figure name' ) ] ),
           ( 'hide-last-ticks',   [ [ '{p}{p}hlt'.format( **fmt ) ], dict( type=int, default=0, help='hide N last ticks' ) ] )
        ] )

    args_included = args_included or arguments.keys()

    if single:
        axes_subparser = SingleSubparser( prefix_chars=prefix_char, prog='axes' )
    else:
        axes_subparser = ArgumentParser( prefix_chars=prefix_char, prog='axes', add_help=False )
    for key in args_included:
        if key in args_excluded:
            continue
        args, kwargs = arguments[key]
        if key in defaults:
            kwargs['default'] = defaults[key]
        axes_subparser.add_argument( prefix_char+prefix_char+key, *args, **kwargs )

    return axes_subparser

