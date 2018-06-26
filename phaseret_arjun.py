#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
List of utilities for phase retrieval

@author: Arjun1
"""
import numpy as np


def my_fft(input):
    return np.fft.fftshift(np.fft.fftn(input))
def my_ifft(input):
    return np.fft.ifftn(np.fft.ifftshift(input))


def convertToPixelPositions(positions,pixelSize,littleArea):
    positions = np.array(positions) / float(pixelSize)
    positions[:,0] = positions[:,0] - np.min(positions[:,0])
    positions[:,1] = positions[:,1] - np.min(positions[:,1])
    positions[:,0] = positions[:,0] - np.round(np.max(positions[:,0]) / 2.0)
    positions[:,1] = positions[:,1] - np.round(np.max(positions[:,1]) / 2.0)
    bigx = littleArea + np.round(np.max(positions)) * 2.0 + 10
    bigy = littleArea + np.round(np.max(positions)) * 2.0 + 10
    #big_cent = np.floor(np.array(bigx) / 2.0) + 1 #wrong center
    bigCent = bigx // 2
    pixelPositions = positions + bigCent
    return pixelPositions, bigx, bigy


def makeCircleMask(radius,imgSize):
#    nc = np.ceil(imgSize / 2.0)
#    n2 = nc - 1
#    nx = np.arange(-n2,n2+1,1)
#    ny = np.arange(-n2,n2+1,1)
#    xx, yy = np.meshgrid(nx,ny) cleaner version below
    nc = imgSize // 2
    nx = np.arange(imgSize) - nc
    ny = np.arange(imgSize) - nc
    xx, yy = np.meshgrid(nx,ny)
    R = np.sqrt(xx**2 + yy**2)
    out = np.zeros([imgSize,imgSize])
    out[R <= radius] = 1
    return out


def getROI(image, centerX, centerY, cropSize):
    bX = image.shape[0]
    bY = image.shape[1]
    halfCropSize = cropSize // 2
##    roi = []
##    for i in range(0,len(centerX)):
#    if np.mod(cropSize, 2) == 0:
#        roi = np.array([np.arange(centerX-halfCropSize,centerX + halfCropSize,1),
#                    np.arange(centerY - halfCropSize,centerY + halfCropSize,1)])
#    else:
#        roi = np.array([np.arange(centerX-halfCropSize,centerX + halfCropSize+1,1),
#                    np.arange(centerY - halfCropSize,centerY + halfCropSize+1,1)])
    bX = image.shape[0]
    bY = image.shape[1]
    halfCropSize = cropSize // 2
    vec = np.arange(cropSize) - halfCropSize
    roi = np.array([vec+centerX,vec+centerY])
    return roi, bX, bY

def rFactor(a,b):
    R = np.sum(np.abs(np.abs(a) - np.abs(b))) / np.sum(np.abs(a))
    return R

def fresnel_advance(u_0, dx, dy, z, llambda):    
    # input a field u_0, spatial resolution dx dy, propagation distance z, 
    #wavelength llambda and outputs a propagated field u_out
    k = 2 * np.pi / float(llambda)
    ny, nx = u_0.shape
    l_x = float(dx) * nx
    l_y = float(dy) * ny
    df_x = 1 / l_x
    df_y = 1 / l_y
    yu_vec = np.ones([nx,1])
    xu_vec = np.arange(nx).reshape(1,nx) - nx // 2
    yv_vec = np.ones([ny,1]).reshape(1,ny)
    xv_vec = np.arange(ny).reshape(ny,1) - ny // 2
    u = yu_vec.dot(xu_vec) * df_x
    v =  xv_vec.dot(yv_vec)* df_y
    
    O = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(u_0)))
    H = np.exp(1j * k * float(z)) * np.exp(-1j*np.pi*float(llambda)*float(z)*( u** 2 + v ** 2))
    u_out = np.fft.fftshift((np.fft.ifftn(np.fft.ifftshift(O * H))))
    return u_out 
    
    
    
    
    
    