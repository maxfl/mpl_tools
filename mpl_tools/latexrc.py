#!/usr/bin/env python
# encoding: utf-8

import matplotlib
matplotlib.rc('font', family='serif', serif=[], monospace=[], **{ 'sans-serif' : [] })
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', unicode=True)
matplotlib.rc('text.latex', preamble=[ r'\IfFileExists{maxfl-loc}{\usepackage[rus]{maxfl-loc}}{\usepackage[russian]{babel}}' ])

matplotlib.rc('pgf', preamble=[ r'\IfFileExists{maxfl-loc}{\usepackage[rus]{maxfl-loc}}{\usepackage[russian]{babel}}' ],
                     texsystem='lualatex', rcfonts=False )

def usepackage( package, options='' ):
    preamble = matplotlib.rcParams['text.latex.preamble']
    matplotlib.rc('text.latex', 
                  preamble=preamble+[ r'\IfFileExists{%s.sty}'
                                      r'{\usepackage%s{%s}}{}'%( 
                                      package, options and '[%s]'%options or '', package )] )

from matplotlib import pyplot as plt
from mpl_tools import savefig
