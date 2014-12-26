#!/usr/bin/env python
# encoding: utf-8

import matplotlib
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':16})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', unicode=True)
matplotlib.rc('text.latex', preamble=[ '\usepackage[russian]{babel}' ])

def usepackage( package, options='' ):
    preamble = matplotlib.rcParams['text.latex.preamble']
    matplotlib.rc('text.latex', 
                  preamble=preamble+['\IfFileExists{%s.sty}'
                                     '{\usepackage%s{%s}}{}'%( 
                                     package, options and '[%s]'%options or '', package )] )

from matplotlib import pyplot as plt
from mpl_tools import savefig
