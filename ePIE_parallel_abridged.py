#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:01:27 2017

@author: Arjun1
"""
#%%
import os
import numpy as np
import scipy.io as sio
import phaseret_arjun as pr
from mpi4py import MPI
import time
import argparse
#%%
job_id = os.environ['JOB_ID']
#%% get parallel size and rank
comm = MPI.COMM_WORLD
size = comm.Get_size() #total number of parallel processors, should
#be equal to length of wavelength vector
rank = comm.Get_rank() #ID number of this instance from 0:size-1
master_rank = 0 #processor where everything is saved
#%% initialize parser for mandatory/optional inputs
parser = argparse.ArgumentParser()
parser.add_argument("-bo","--beta_obj",type=float,default=1.0,help="beta for object update")
parser.add_argument("-ba", "--beta_ap",type=float,default=1.0,help="beta for probe update")
parser.add_argument("-ns","--no_save",action="store_true",help="don't save output")
parser.add_argument("-mn","--misc_notes",default='',help="miscellaneous notes")
parser.add_argument("-i","--input_file",help="full/path/to/input")
args = parser.parse_args()
#%% loading inputs
input_file = args.input_file
mat_contents = sio.loadmat(input_file, struct_as_record=False)
ePIE_struct = mat_contents['ePIE_inputs']
diffpats = ePIE_struct[0,0].Patterns
positions = ePIE_struct[0,0].Positions 
file_name = ePIE_struct[0,0].FileName
pixel_size = np.squeeze(ePIE_struct[0,0].PixelSize)
big_obj = np.squeeze(ePIE_struct[0,0].InitialObj)
ap_radius = np.squeeze(ePIE_struct[0,0].ApRadius)
aperture = np.squeeze(ePIE_struct[0,0].InitialAp)
iterations = np.squeeze(ePIE_struct[0,0].Iterations)
show_im = 0
s = np.squeeze(ePIE_struct[0,0].S)
s = s[:,0]
s_true = np.squeeze(ePIE_struct[0,0].S_true)
s_true = s_true[:,0]
n_modes = pixel_size.shape[0]
del(mat_contents)
del(ePIE_struct)

if size != n_modes:
    raise Exception('Number of workers must equal number of modes') 
#%% reconstruction parameters
beta_ap = args.beta_ap
beta_obj = args.beta_obj
updateAperture = 0
freezeAperture = 10000
save_flag = bool(not(args.no_save))
#%% print parameters to output file
if rank == 0:
    print "==================broadband ePIE parameters=================="
    print "input: %s" % input_file
    print "output: %s" % file_name
    print "iterations: %d" % iterations
    print "beta obj: %r" % beta_obj
    print "beta_ap: %r" % beta_ap
    print "number of modes: %d" % n_modes
    print "misc notes: %s" % args.misc_notes
#%% define parameters from data and for reconstruction
for i in range(diffpats.shape[2]):
    diffpats[:,:,i] = np.fft.fftshift(diffpats[:,:,i])
    
yKSpace = [diffpats.shape[0], diffpats.shape[1]]
nApert = diffpats.shape[2]
bestErr = 100
littleArea = yKSpace[0] #dp should be square
littleCent = littleArea // 2
cropVec = np.arange(littleArea) - littleCent
#%% getting center positions for cropping ROIs (parallel for each wavelength)
positionArrayLocal, bigXLocal, bigYLocal = (pr.convertToPixelPositions(positions,
                                                pixel_size[rank],littleArea))
centerYLocal = np.round(positionArrayLocal[:,1]).astype(int)
centerXLocal = np.round(positionArrayLocal[:,0]).astype(int)
bigXLocal = bigXLocal.astype(int)
centBig = bigXLocal // 2
bigYLocal = bigYLocal.astype(int)
cropR = np.zeros([nApert,littleArea], dtype=int)
cropC = np.zeros([nApert, littleArea], dtype=int)
for aper in range(nApert):
    cropR[aper,:] = cropVec + centerYLocal[aper]
    cropC[aper,:] = cropVec + centerXLocal[aper] 
#%% create initial guess for aperture and object

if aperture[rank] == 0:
    aperture = pr.makeCircleMask(np.ceil(ap_radius / pixel_size[rank]),littleArea)
    initialAperture = aperture.copy()
else:
    initialAperture = aperture.copy()  

if big_obj[rank] ==  0:                  
    big_obj = (np.random.rand(bigXLocal, bigYLocal) * 
                  np.exp(1j * np.random.rand(bigXLocal, bigYLocal)))
    initialObj = big_obj.copy()
else:
    initialObj = big_obj.copy()
aperture = aperture + 0j
fourierErrorGlobal = np.zeros([iterations,nApert])      
 #%% main reconstruction loop

for itt in range(iterations):
    t0 = time.time()
    if rank == 0:
        posOrder = np.random.permutation(np.arange(nApert,dtype='i'))
    else:
        posOrder = np.empty(nApert,dtype='i')
    comm.Bcast(posOrder, root=0) #broadcasting the same permutation order
    #to all modes
    for aper in posOrder:
        currentDP = diffpats[:,:,aper].copy()
        rSpace = big_obj[cropR[aper,:]][:,cropC[aper,:]].copy()
        bufferRSpace = rSpace.copy()
        objectMax = np.max(np.abs(rSpace))
        probeMax = np.max(np.abs(aperture))
        weight = np.sqrt(s[rank]) / np.sqrt(np.sum(np.abs(aperture)**2))
        bufferExitWave = weight * rSpace * aperture
        updateExitWave = bufferExitWave.copy()
        tempDP = np.fft.fft2(updateExitWave)
        collectedMags = np.empty([yKSpace[0],yKSpace[1]])
        comm.Allreduce(np.abs(tempDP) ** 2, collectedMags, op=MPI.SUM)
        collectedMags = collectedMags.T #for some reason, Allreduce transposes in this context
        tempDP = np.sqrt(currentDP) * tempDP / np.sqrt(collectedMags)
        newExitWave = np.fft.ifft2(tempDP)
        diffExitWave = newExitWave - bufferExitWave
        updateFactorObj = np.conj(aperture) / probeMax ** 2
        newRSpace = bufferRSpace + updateFactorObj * beta_obj * diffExitWave       
        big_obj[np.ix_(cropR[aper,:],cropC[aper,:])] = newRSpace.copy()
        updateFactorPr = beta_ap / objectMax ** 2
        aperture += updateFactorPr * np.conj(bufferRSpace) * diffExitWave            
        s[rank] = np.sum(np.abs(aperture)**2)
        fourierErrorGlobal[itt,aper] = (np.sum(np.abs(np.sqrt(currentDP) - np.sqrt(collectedMags))) / 
                          np.sum(np.sqrt(currentDP)))
        
    meanErr = np.mean(fourierErrorGlobal[itt,:]) 
    if rank == 0:
        print "%d. %r [%r sec]" % (itt, meanErr, time.time()-t0)
    if bestErr > meanErr:
        bestObj = big_obj.copy()
        bestErr = meanErr.copy()
    comm.barrier()
#%%gather S
if rank == master_rank:
    sGlobal = np.empty([n_modes,1])
else:
    sGlobal = None    
comm.Gather(s[rank], sGlobal, root=master_rank)    
#%% gather and save
if save_flag:  
    bestObjs = {}
    apertures = {}
    bestObjs = comm.gather(bestObj,root=master_rank)
    apertures = comm.gather(aperture, root=master_rank)
    if rank == master_rank:
        saveString = "ePIE_output_%s" % job_id
        np.savez(saveString,bestObj=bestObjs, aperture=apertures,
                 fourierError=fourierErrorGlobal, s=sGlobal) #save output as dict 















