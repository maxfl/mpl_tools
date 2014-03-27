#!/usr/bin/env python
# encoding: utf-8

import ROOT, numpy

def get_buffers_tree( tree, expr, cut='' ):
    """Fill and return a pointer to the TTree buffers. Should be copied"""
    count = tree.Draw( expr, cut, 'goff' )
    n = expr.count(':') + 1
    assert n<5, 'Incompartible number of buffers '+str(n)
    if count<=0: return [ None ]*n
    buffers = [ tree.GetV1, tree.GetV2, tree.GetV3, tree.GetV4 ]
    return ( numpy.frombuffer( buf(), dtype='d', count=count ) for buf in buffers[:n] )
##end def get_buffers_tree

def get_buffers_graph( g ):
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints==0: return None, None
    from numpy import frombuffer, double
    return frombuffer( g.GetX(), double, npoints)\
         , frombuffer( g.GetY(), double, npoints)
##end def function

def get_err_buffers_graph( g ):
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints==0: return None, None
    from numpy import frombuffer, double
    return frombuffer( g.GetEX(), double, npoints)\
         , frombuffer( g.GetEY(), double, npoints)
##end def function

def get_err_buffers_graph_asymm( g ):
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints==0: return None, None, None, None
    from numpy import frombuffer, double
    return frombuffer( g.GetEXlow(), double, npoints)\
         , frombuffer( g.GetEXhigh(), double, npoints)\
         , frombuffer( g.GetEYlow(), double, npoints)\
         , frombuffer( g.GetEYhigh(), double, npoints)
##end def function

def convert_hist1( h, cls, newname=None, addbins=None, multbins=None, noerr=False ):
    """
    Convert TH1X to TH1Y
    addbins=(n, x2a) will add n bins to the end of the histogram increasing x2 to x2a
                     new bins will be empty
    multbins=n will rebin the resulting histogram increasing the number of bins n times
             each bin will be split in n bins with n times smaller height
             and n times smaller err^2
    addbins and multbins can not be applied in the same time
    """
    ax = h.GetXaxis()
    xbins = ax.GetXbins()
    newh = None
    if newname==None: newname=h.GetName()
    n = h.GetNbinsX()
    if xbins.GetSize()==0:
        x1, x2 = ax.GetXmin(), ax.GetXmax()
        if addbins:
            assert multbins==None, 'Can not add/mult bins in the same time'
            na, x2a = addbins
            newh = cls( newname, h.GetTitle(), na, x1, x2a )
        elif multbins:
            na = n*multbins
            newh = cls( newname, h.GetTitle(), na, x1, x2 )
        else:
            newh = cls( newname, h.GetTitle(), n, x1, x2 )
        ##end if
    else:
        assert addbins==None, 'Can not add bins to variable bins histogram'
        newh = cls( newname, h.GetTitle(), h.GetNbinsX(), xbins.GetArray() )
    ##end if
    err = get_err_buffer_hist1( h, flows=True ) if not noerr else None
    b = get_buffer_hist1( h, flows=True )
    newb = get_buffer_hist( newh, flows=True )

    if multbins:
        if err!=None:
            newh.Sumw2()
            newerr = get_err_buffer_hist1( newh, flows=True )
            newerr[0], newerr[-1] = err[0], err[-1]
            newerr1=newerr[1:-1].reshape( ( multbins, n ), order='F' )
            newerr1[:]=err[1:-1]/float(multbins)
        ##end if err

        newb[0], newb[-1] = b[0], b[-1]
        newb1=newb[1:-1].reshape( ( multbins, n ), order='F' )
        newb1[:]=b[1:-1]/float(multbins)
    else:
        if err!=None:
            newh.Sumw2()
            newerr = get_err_buffer_hist1( newh, flows=True )
            newerr[:n+2] = err[:]
        ##end if err

        newb[:n+2] = b[:]
    ##end if multbins

    newh.SetEntries( h.GetEntries() )

    return newh
##end def histConvert

def get_buffer_vector( self ):
    from numpy import frombuffer, dtype
    buf = self.GetMatrixArray()
    return frombuffer( buf, dtype( buf.typecode ), self.GetNoElements() )
##end def get_buffer_vector

def get_buffer_array( self ):
    from numpy import frombuffer, dtype
    buf = self.GetArray()
    return frombuffer( buf, dtype( buf.typecode ), self.GetSize() )
##end def get_buffer_vector

def get_buffer_matrix( m ):
    from numpy import frombuffer, dtype
    cbuf = m.GetMatrixArray()
    buf = frombuffer( cbuf, dtype( cbuf.typecode ), m.GetNoElements() )
    return buf.reshape( m.GetNrows(), m.GetNcols() )
##end def

def get_buffer_hist1( h, flows=False ):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    """
    from numpy import frombuffer, dtype
    buf = h.GetArray()
    buf = frombuffer(buf , dtype( buf.typecode ), h.GetNbinsX()+2 )
    if not flows: buf = buf[1:-1]
    return buf
##end def get_buffer_hist

def get_err_buffer_hist1( h, flows=False ):
    """Return histogram error buffer
    if flows=False, exclude underflow and overflow
    """
    sw2 = h.GetSumw2()
    if sw2.GetSize()==0: return None

    from numpy import frombuffer, double
    buf = frombuffer( sw2.GetArray(), double, h.GetNbinsX()+2 )
    if not flows: buf = buf[1:-1]
    return buf
##end def get_err_buffer_hist1

def get_buffer_hist2( h, flows=False, mask=None ):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    if mask=0.0 than bins with 0.0 content will be white, but not colored
    NOTE: buf[biny][binx] is the right access signature
    """
    from numpy import frombuffer, dtype
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = h.GetArray()
    buf = frombuffer( buf, dtype( buf.typecode ), (nx+2)*(ny+2) ).reshape( ( ny+2, nx+2 ) )
    if mask!=None:
        from numpy import ma
        buf = ma.array( buf, mask = buf==mask )
    ##end if mask!=None
    if flows: return buf

    buf = buf[1:ny+1,1:nx+1]
    return buf
##end def get_buffer_hist

def get_err_buffer_hist2( h, flows=False ):
    """Return histogram error buffer
    if flows=False, exclude underflow and overflow
    if mask=0.0 than bins with 0.0 content will be white, but not colored
    NOTE: buf[biny][binx] is the right access signature
    """
    sw2 = h.GetSumw2()
    if sw2.GetSize()==0: return None
    from numpy import frombuffer, dtype
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = sw2.GetArray()
    buf = frombuffer( buf, dtype( buf.typecode ), (nx+2)*(ny+2) ).reshape( ( ny+2, nx+2 ) )
    if flows: return buf
    return buf[1:ny+1,1:nx+1]
##end def get_buffer_hist

def get_buffer_histN( h, flows=False, mask=None, err=False ):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    if mask=0.0 than bins with 0.0 content will be white, but not colored
    NOTE: buf[biny][binx] is the right access signature
    """
    ndim = h.GetDimension()
    shape = []
    n = 1
    assert ndim<4
    if ndim==3:
        nz = h.GetNbinsZ()+2
        n*=nz
        shape.append( nz )
    ##end if
    if ndim>=2:
        ny = h.GetNbinsY()+2
        n*=ny
        shape.append( ny )
    ##end if
    if ndim>=1:
        nx = h.GetNbinsX()+2
        n*=nx
        shape.append( nx )
    ##end if
    from numpy import frombuffer, dtype
    arr = err and h.GetSumw2() or h
    if arr.GetSize()==0: return None
    buf = arr.GetArray()
    buf = frombuffer( buf, dtype( buf.typecode ), n ).reshape( shape )
    if mask!=None:
        from numpy import ma
        buf = ma.array( buf, mask = buf==mask )
    ##end if mask!=None
    if flows: return buf

    if ndim==3: return buf[1:-1,1:-1,1:-1]
    if ndim==2: return buf[1:-1,1:-1]
    return buf[1:-1]
##end def get_buffer_hist

def get_buffer_hist( h, flows=False, mask=None ):
    return get_buffer_histN( h, flows, mask, err=False )
##end def get_buffer_histN

def get_err_buffer_hist( h, flows=False ):
    return get_buffer_histN( h, flows, err=True )
##end def get_buffer_histN

def get_buffers_graph2d( g ):
    n = g.GetN()
    if n==0: return None, None, None
    from numpy import frombuffer, double
    x = frombuffer( g.GetX(), double, n )
    y = frombuffer( g.GetY(), double, n )
    z = frombuffer( g.GetZ(), double, n )
    return x, y, z
##end def tget_buffers_graph2d

def get_buffers_mat_graph2d( g, shape ):
    x, y, z = get_buffers_graph2d( g )
    if x==None:
        return None, None, None
    return x.reshape( shape ), y.reshape( shape ), z.reshape( shape )
##end def tget_buffers_graph2d

def get_bin_edges_axis( ax, type=False, rep=None ):
    """Get the array with bin edges"""
    import numpy
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    lims=None
    fixed = False
    if n>0:
        from numpy import frombuffer, double
        lims = frombuffer( xbins.GetArray(), double, n )
        fixed = False
    else:
        from numpy import linspace
        lims = linspace( ax.GetXmin(), ax.GetXmax(), ax.GetNbins()+1 )
        fixed = True
    ##end if
    if rep and rep>1:
        res = [ lims ]
        delta = -lims[0]
        for i in xrange( rep-1 ):
            res.append( res[-1][-1] + lims[1:] + delta )
        ##end for i
        lims = numpy.concatenate( res )
    ##end if rep


    if type: return lims, fixed
    return lims
##end def

def get_bin_centers_axis( ax ):
    """Get the array with bin centers"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n>0:
        from numpy import frombuffer, double
        lims = frombuffer( xbins.GetArray(), double, n )
        return ( lims[:-1] + lims[1:] )*0.5
    ##end if
    import numpy
    hwidth = ax.GetBinWidth(1)*0.5
    return numpy.linspace( ax.GetXmin()+hwidth, ax.GetXmax()-hwidth, ax.GetNbins() )
##end def

def get_bin_widths_axis( ax ):
    """Get the array with bin widths or bin width if it's constant"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n>0:
        from numpy import frombuffer, double
        lims = frombuffer( xbins.GetArray(), double, n )
        return ( lims[1:] - lims[:-1] )
    ##end if
    return ax.GetBinWidth(1)
##end def

def bind_functions():
    setattr( ROOT.TArray, 'get_buffer', get_buffer_array )

    setattr( ROOT.TAxis, 'get_bin_centers', get_bin_centers_axis )
    setattr( ROOT.TAxis, 'get_bin_edges', get_bin_edges_axis )
    setattr( ROOT.TAxis, 'get_bin_widths', get_bin_widths_axis )

    setattr( ROOT.TGraph, 'get_buffers', get_buffers_graph )
    setattr( ROOT.TGraphErrors, 'get_err_buffers', get_err_buffers_graph )
    setattr( ROOT.TGraphAsymmErrors, 'get_err_buffers', get_err_buffers_graph_asymm )

    setattr( ROOT.TGraph2D, 'get_buffers', get_buffers_graph2d )
    setattr( ROOT.TGraph2D, 'get_buffers_mat', get_buffers_mat_graph2d )

    setattr( ROOT.TH1, 'convert', convert_hist1 )
    setattr( ROOT.TH1, 'get_buffer', get_buffer_hist1 )
    setattr( ROOT.TH1, 'get_err_buffer', get_err_buffer_hist1 )

    setattr( ROOT.TH2, 'get_buffer', get_buffer_hist2 )
    setattr( ROOT.TH2, 'get_err_buffer', get_err_buffer_hist2 )
    setattr( ROOT.TH3, 'get_buffer', get_buffer_hist )
    setattr( ROOT.TH3, 'get_err_buffer', get_err_buffer_hist )

    setattr( ROOT.TMatrixD, 'get_buffer', get_buffer_matrix )
    setattr( ROOT.TMatrixF, 'get_buffer', get_buffer_matrix )

    setattr( ROOT.TTree, 'get_transient_buffers', get_buffers_tree )

    setattr( ROOT.TVectorD, 'get_buffer', get_buffer_vector )
    setattr( ROOT.TVectorF, 'get_buffer', get_buffer_vector )
##end def bind_functions
