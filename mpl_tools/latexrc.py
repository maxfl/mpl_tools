#!/usr/bin/env python
# encoding: utf-8

import matplotlib
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':16})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', unicode=True)
matplotlib.rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
matplotlib.rc('text.latex', preamble='\usepackage[english]{babel}')

from matplotlib import pyplot as plt
from mpl_tools import savefig
