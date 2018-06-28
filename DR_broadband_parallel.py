#!/u/local/apps/python/2.7.13/bin/python
# -*- coding: utf-8 -*-
"""
most updated version with relaxation term
@author: Arjun Rana
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
#%% initialize parser for mandatory/optional inputs
parser = argparse.ArgumentParser()
parser.add_argument("-bo","--beta_obj",type=float,default=0.9,help="beta for object update")
parser.add_argument("-ba", "--beta_ap",type=float,default=0.9,help="beta for probe update")
parser.add_argument("-pn","--probe_norm",action="store_true",help="probe normalization flag")
parser.add_argument("-al", "--alpha",type=float,default=0.1,help="relaxation term")
parser.add_argument("-sip","--semi_implicit_P",action="store_true",help="semi implicit probe update flag")
parser.add_argument("-wi", "--weight_initial",type=float,default=0.1,help="initial weight for fourier constraint")
parser.add_argument("-wf", "--weight_final",type=float,default=0.6,help="final weight for fourier constraint")
parser.add_argument("-or", "--order",type=float,default=6,help="order of weight")
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
#s = s[:,0]
s_true = np.squeeze(ePIE_struct[0,0].S_true)
#s_true = s_true[:,0]
n_modes = pixel_size.shape[0]
del(mat_contents)
del(ePIE_struct)

if size != n_modes:
    raise Exception('Number of workers must equal number of modes') 
#%% reconstruction parameters
beta_ap = args.beta_ap
beta_obj = args.beta_obj
probe_norm = bool(args.probe_norm)
alpha = args.alpha
semi_implicit_P = bool(args.semi_implicit_P)
weight_initial = args.weight_initial
weight_final = args.weight_final
order = args.order
updateAperture = 0
freezeAperture = float('inf')
save_flag = bool(not(args.no_save))
#%% print parameters to output file
if rank == 0:
    print "==================broadband DR parameters=================="
    print "input: %s" % input_file
    print "output: %s" % file_name
    print "iterations: %d" % iterations
    print "beta obj: %r" % beta_obj
    print "beta_ap: %r" % beta_ap
    print "probe norm: %r" % probe_norm
    print "alpha: %r" % alpha
    print "semi_implicit_P: %r" % semi_implicit_P
    print "weight_initial: %r" % weight_initial
    print "weight_final: %r" % weight_final
    print "number of modes: %d" % n_modes
    print "misc notes: %s" % args.misc_notes
#%% define parameters from data and for reconstruction
for i in range(diffpats.shape[2]):
    diffpats[:,:,i] = np.fft.fftshift(np.sqrt(diffpats[:,:,i]))
    
y_kspace = [diffpats.shape[0], diffpats.shape[1]]
nApert = diffpats.shape[2]
bestErr = 100
littleArea = y_kspace[0] #dp should be square
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
Z = np.zeros([y_kspace[0],y_kspace[1],nApert],dtype=complex)
ws = weight_initial + (weight_final - weight_initial)* ((np.arange(iterations,dtype=float)+1)/iterations) ** order
alpha_itts = alpha - (alpha-0.1) * ((np.arange(iterations,dtype=float)+1)/iterations) ** 2.0
fourierErrorGlobal = np.zeros([iterations,nApert]) 
     
 #%% main reconstruction loop
if rank == 0: 
    print '=============starting DR reconstruction============='
for itt in range(iterations):
    t0 = time.time()
    w = ws[itt]
    alpha_itt = alpha_itts[itt].copy()
    if rank == 0:
        posOrder = np.random.permutation(np.arange(nApert,dtype='i'))
    else:
        posOrder = np.empty(nApert,dtype='i')
    comm.Bcast(posOrder, root=0) #broadcasting the same permutation order
    #to all modes
    for aper in posOrder:
        current_dp = diffpats[:,:,aper].copy()
        u_old = big_obj[cropR[aper,:]][:,cropC[aper,:]].copy()
        probe_max = np.max(np.abs(aperture))
        p_u = u_old * aperture
        z_0 = np.fft.fft2(p_u)
#        z = Z[:,:,aper].copy()
        z_F = (1+alpha_itt)*z_0 - alpha_itt * Z[:,:,aper]
#        weight = np.sqrt(s[rank]) / np.sqrt(np.sum(np.abs(aperture)**2))
        collected_mags = np.empty([y_kspace[0],y_kspace[1]])
        comm.Allreduce(np.abs(z_F) ** 2, collected_mags, op=MPI.SUM)
#        collected_mags = collected_mags.T #for some reason, Allreduce transposes in this context
        collected_mags = np.sqrt(collected_mags)
        scale = (1-w) * current_dp / collected_mags + w
        z_F = scale * z_F
        z = z_F + alpha_itt * (Z[:,:,aper] - z_0)
        Z[:][:,:,aper] = z.copy()
        p_u_new = np.fft.ifft2(z)
        diff_exit_wave = p_u_new - p_u
        dt = beta_obj / probe_max ** 2
        #update object
        u_new = (((1-beta_obj)/dt) * u_old + p_u_new * np.conj(aperture)) / ((1-beta_obj)/dt + np.abs(aperture)**2)
        big_obj[np.ix_(cropR[aper,:],cropC[aper,:])] = u_new.copy()
        object_max = np.max(np.abs(u_new))
        ds = beta_ap / object_max ** 2
        #update probe
        aperture = (((1-beta_ap)*aperture + ds * p_u_new * np.conj(u_new)) / 
                    ((1-beta_ap) + ds * np.abs(u_new) ** 2))            
        s[rank] = np.sum(np.abs(aperture)**2)
        fourierErrorGlobal[itt,aper] = (np.sum(np.abs(current_dp - collected_mags)) / 
                          np.sum(current_dp))
        
    meanErr = np.mean(fourierErrorGlobal[itt,:]) 
    if rank == 0:
        print "%d. %r [%r sec]" % (itt, meanErr, time.time()-t0)
    if bestErr > meanErr:
        bestObj = big_obj.copy()
        bestErr = meanErr.copy()
    if np.mod(itt,10) == 0:
        if rank == 0:
            sGlobal = np.empty([n_modes,1])
        else:
            sGlobal = None    
        comm.Gather(s[rank], sGlobal, root=0)
        bestObjs = {}
        apertures = {}
        bestObjs = comm.gather(bestObj,root=0)
        apertures = comm.gather(aperture, root=0)
        if rank == 0:
            saveString = "DR_output_itt_%s_%s" % (itt, job_id)
            np.savez(saveString,bestObj=bestObjs, aperture=apertures,
                 fourierError=fourierErrorGlobal, s=sGlobal)
    comm.barrier()
#%%gather S
if rank == 0:
    sGlobal = np.empty([n_modes,1])
else:
    sGlobal = None    
comm.Gather(s[rank], sGlobal, root=0)    
#%% gather and save
if save_flag:  
    bestObjs = {}
    apertures = {}
    bestObjs = comm.gather(bestObj,root=0)
    apertures = comm.gather(aperture, root=0)
    if rank == 0:
        saveString = "DR_output_%s" % job_id
        np.savez(saveString,bestObj=bestObjs, aperture=apertures,
                 fourierError=fourierErrorGlobal, s=sGlobal) #save output as dict 
if rank == 0:
    print 'reconstruction complete!'













