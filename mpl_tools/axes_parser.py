#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from argparse_utils import SingleSubparser
from argparse import ArgumentParser

def ApplyAxesOptions( opts, ax=None ):
    if not opts: 
        return
    ax = ax or plt.gca()
    ax.grid( opts.grid )
    if opts.title:
        ax.set_title( opts.title )
    if opts.x_lim:
        ax.set_xlim( opts.x_lim )
    if opts.y_lim:
        ax.set_ylim( opts.y_lim )
    if opts.z_lim:
        ax.set_zlim( opts.z_lim )

    if opts.legend:
        leg = ax.legend( loc=opts.legend_location, title=Translate(opts.legend_title) )

def AxesSubparser( prefix_char='+', args_included=[], args_excluded=[], defaults={} ):
    arguments = dict( [
           #( 'labels',          [ [ '%sl'%prefix_char ], dict( nargs='+', help='labels' ) ] ),
           ( 'title',           [ [ '%st'%prefix_char ], dict( default='', help='title' ) ]  ),
           ( 'no-grid',         [ [ '%sG'%prefix_char ], dict( action='store_false', dest='grid', help='no grid' ) ] ),
           ( 'grid',            [ [ '%sg'%prefix_char ], dict( action='store_true', help='grid' ) ] ),
           ( 'legend',          [ [ '%sl'%prefix_char ], dict( action='store_true', help='legend' )] ),
           ( 'legend-title',    [ [], dict( help='legend title' )] ),
           ( 'legend-location', [ [ '%sL'%prefix_char ], dict( default='upper right', help='legend location') ]  ),
           ( 'x-lim',           [ [ '%sx'%prefix_char ], dict( type=float, nargs=2, help='x limits') ] ),
           ( 'y-lim',           [ [ '%sy'%prefix_char ], dict( type=float, nargs=2, help='y limits') ] ),
           ( 'z-lim',           [ [ '%sz'%prefix_char ], dict( type=float, nargs=2, help='z limits') ] ),
        ] )

    args_included = args_included or arguments.keys()

    #axes_subparser = SingleSubparser( prefix_chars=prefix_char, prog='axes' )
    axes_subparser = ArgumentParser( prefix_chars=prefix_char, prog='axes', add_help=False )
    for key in args_included:
        if key in args_excluded:
            continue
        args, kwargs = arguments[key]
        if key in defaults:
            kwargs['default'] = defaults[key]
        axes_subparser.add_argument( prefix_char+prefix_char+key, *args, **kwargs )

    return axes_subparser

