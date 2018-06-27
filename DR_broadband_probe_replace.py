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
parser.add_argument("-raa", "--replace_ap_after",type=int,default=0,help="when to start probe replacement")
parser.add_argument("-rau", "--replace_ap_until",type=int,default=1000000000,help="when to stop probe replacement")
parser.add_argument("-arf", "--ap_repl_freq",type=float,default=0.1,help="probe replacement frequency")
parser.add_argument("-bm","--best_mode", type=int, default=100000,help="optional input best mode if not in ePIE_inputs")
parser.add_argument("-fd","--fresnel_dist", type=float, default=0.0,help="optional input fresnel distance if not in ePIE_inputs")
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
llambda = np.squeeze(ePIE_struct[0,0].__getattribute__('lambda'))
show_im = 0
s = np.squeeze(ePIE_struct[0,0].S)
s_true = np.squeeze(ePIE_struct[0,0].S_true)
n_modes = pixel_size.shape[0]
try:
    best_mode = int(np.squeeze(ePIE_struct[0,0].central_mode) - 1) #to account for difference in indexing
except AttributeError:
    best_mode = args.best_mode
try:    
    fresnel_dist = float(np.squeeze(ePIE_struct[0,0].fresnel_dist))
except AttributeError:
    fresnel_dist = args.fresnel_dist

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
replace_ap_after = args.replace_ap_after
replace_ap_until = args.replace_ap_until
ap_repl_freq = args.ap_repl_freq
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
    print "probe replacement frequency: %r" % ap_repl_freq
    print "probe replacement after iteration: %d" % replace_ap_after
    print "probe replacement until iteration: %d" % replace_ap_until
    print "best mode: %d" % best_mode
    print "fresnel distance: %r" % fresnel_dist
    print "misc notes: %s" % args.misc_notes
#%% define parameters from data and for reconstruction
for i in range(diffpats.shape[2]):
    diffpats[:,:,i] = np.fft.fftshift(np.sqrt(diffpats[:,:,i]))
    
y_kspace = [diffpats.shape[0], diffpats.shape[1]]
n_apert = diffpats.shape[2]
bestErr = 100
little_area = y_kspace[0] #dp should be square
little_cent = little_area // 2
cropVec = np.arange(little_area) - little_cent
#%% getting center positions for cropping ROIs (parallel for each wavelength)
positionArrayLocal, bigXLocal, bigYLocal = (pr.convertToPixelPositions(positions,
                                                pixel_size[rank],little_area))
centerYLocal = np.round(positionArrayLocal[:,1]).astype(int)
centerXLocal = np.round(positionArrayLocal[:,0]).astype(int)
bigXLocal = bigXLocal.astype(int)
centBig = bigXLocal // 2
bigYLocal = bigYLocal.astype(int)
cropR = np.zeros([n_apert,little_area], dtype=int)
cropC = np.zeros([n_apert, little_area], dtype=int)
for aper in range(n_apert):
    cropR[aper,:] = cropVec + centerYLocal[aper]
    cropC[aper,:] = cropVec + centerXLocal[aper] 
#%% create initial guess for aperture and object

if aperture[rank] == 0:
    aperture = pr.makeCircleMask(np.ceil(ap_radius / pixel_size[rank]),little_area)
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
Z = np.zeros([y_kspace[0],y_kspace[1],n_apert],dtype=complex)
ws = weight_initial + (weight_final - weight_initial)* ((np.arange(iterations,dtype=float)+1)/iterations) ** order
alpha_itts = alpha - (alpha-0.1) * ((np.arange(iterations,dtype=float)+1)/iterations) ** 2.0
fourierErrorGlobal = np.zeros([iterations,n_apert]) 

#%% probe replacement parameters and allocation of propagators
scaling_ratio = pixel_size / pixel_size[best_mode] 
scoop_size = int(np.round(little_area/scaling_ratio[rank]))
scoop_center = scoop_size // 2
scoop_vec = np.arange(scoop_size) - scoop_center + little_cent    
scoop_range = scoop_vec[-1] - scoop_vec[0] + 1
if scoop_range > little_area:
    pad_pre = int(np.ceil((scoop_range-little_area)/2.0))
    pad_post = int(np.float((scoop_range-little_area)/2.0))
else:
    pad_pre = 0
    pad_post = 0
cutoff = iterations / 2
prb_rplment_weight = np.squeeze(np.minimum([(cutoff**4.0/10.0) / (np.arange(iterations)+1)**4.0], [0.1]))

k = 2 * np.pi * llambda[rank]
l_x = pixel_size[rank] * little_area
l_y = pixel_size[rank] * little_area
df_x = 1 / l_x
df_y = 1 / l_y
yu_vec = np.ones([little_area,1])
xu_vec = np.arange(little_area).reshape(1,little_area) - little_area // 2
yv_vec = np.ones([little_area,1]).reshape(1,little_area)
xv_vec = np.arange(little_area).reshape(little_area,1) - little_area // 2
u = yu_vec.dot(xu_vec) * df_x
v =  xv_vec.dot(yv_vec)* df_y
if rank != best_mode:
    H_fwd = np.fft.ifftshift(np.exp(1j * k * fresnel_dist) * 
            np.exp(-1j*np.pi*llambda[rank]*fresnel_dist*(u** 2 + v ** 2)))
    H_bk = np.exp(1j * k * -fresnel_dist) * np.exp(-1j*np.pi*llambda[best_mode]
                                            *-fresnel_dist*( u** 2 + v ** 2))
elif rank == best_mode:
    H_bk = np.exp(1j * k * -fresnel_dist) * np.exp(-1j*np.pi*llambda[rank]
                                            *-fresnel_dist*( u** 2 + v ** 2))

#%% pre-allocation of random variables to reduce broadcasting inside loop
if rank == 0:
    rand_seed = np.array(np.random.randint(1000000),ndmin=1)
else:
    rand_seed = np.empty(1,dtype='int64')
comm.Bcast(rand_seed, root=0)
rand_seed = rand_seed[0]
np.random.seed(rand_seed) #all workers have the same RandomState
pos_order = {}
does_ap_replace = {}

for n in range(iterations):
    pos_order[n] = np.random.permutation(np.arange(n_apert,dtype='i'))
    does_ap_replace[n] = np.random.random_sample(n_apert)
     
 #%% main reconstruction loop
if rank == 0: 
    print '=============starting DR(with probe replacement) reconstruction============='
for itt in range(iterations):
    t0 = time.time()
    w = ws[itt]
    alpha_itt = alpha_itts[itt].copy()

    for aper in pos_order[itt]:
        #note: the flag can be allocated outside of the loop to save time
        if (does_ap_replace[itt][aper] <= ap_repl_freq and itt >= replace_ap_after 
            and itt <= replace_ap_until):
            probe_replacement_flag = True
        else:
            probe_replacement_flag = False
            
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
        
        #probe update and replacement
        ap_temp_updated = (((1-beta_ap)*aperture + ds * p_u_new * np.conj(u_new)) / 
                    ((1-beta_ap) + ds * np.abs(u_new) ** 2))
        if probe_replacement_flag:
            #transmit best aperture to all other modes
            if rank != best_mode:
                F_best_mode = np.empty([little_area,little_area],dtype=complex, order='F')
            else:
                F_best_mode = pr.my_fft(aperture) * H_bk
            comm.Bcast(F_best_mode, root=best_mode) 

            if pixel_size[rank] < pixel_size[best_mode]:
                F_probe_replaced = np.pad(F_best_mode, [(pad_pre, pad_post), (pad_pre, pad_post)], mode='constant')
                probe_replaced = pr.my_ifft(F_probe_replaced)
                probe_replaced = probe_replaced[pad_pre:-pad_post][:,pad_pre:-pad_post]
                probe_replaced = np.fft.ifftn(np.fft.fftn(probe_replaced) * H_fwd)
            elif pixel_size[rank] > pixel_size[best_mode]:
                F_best_probe_cropped = F_best_mode[scoop_vec][:,scoop_vec]
                probe_replaced = np.zeros([little_area, little_area], dtype=complex)
                probe_replaced[np.ix_(scoop_vec, scoop_vec)] = pr.my_ifft(F_best_probe_cropped)
                probe_replaced = np.fft.ifftn(np.fft.fftn(probe_replaced) * H_fwd)
            else:
                probe_replaced = ap_temp_updated
            ap_buffer = ap_temp_updated + prb_rplment_weight[itt] * (probe_replaced - ap_temp_updated)
            aperture = np.linalg.norm(ap_temp_updated,'fro')/np.linalg.norm(ap_buffer,'fro') * ap_buffer
        else:
            aperture = ap_temp_updated.copy() 
              
        s[rank] = np.sum(np.abs(aperture)**2)
        fourierErrorGlobal[itt,aper] = (np.sum(np.abs(current_dp - collected_mags)) / 
                          np.sum(current_dp))
        
    meanErr = np.mean(fourierErrorGlobal[itt,:]) 
    if rank == 0:
        print "%d. %r [%r sec]" % (itt, meanErr, time.time()-t0)
    if bestErr > meanErr:
        bestObj = big_obj.copy()
        bestErr = meanErr.copy()
    if (np.mod(itt,50) == 0 and itt > 0):
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
            saveString = "DR_output_probe_replace_itt_%s" % itt
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
        saveString = "DR_output_probe_replace_%s" % job_id
        np.savez(saveString,bestObj=bestObjs, aperture=apertures,
                 fourierError=fourierErrorGlobal, s=sGlobal) #save output as dict 
if rank == 0:
    print 'reconstruction complete!'













