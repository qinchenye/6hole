import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util
import lanczos

def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif z2==z1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
    elif orb1 in pam.O_orbs and orb2 in pam.Obilayer_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]            
    elif orb1 in pam.Obilayer_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1] 
        
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];    
    s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];    
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
    tlabel = slabel[10:15] + [s4,orb4,x4,y4,z4]
    tmp34 = reorder_z(tlabel)
    
    if tmp34 == tlabel:
        slabel2 = slabel + [s4,orb4,x4,y4,z4]
    elif  tmp34 != tlabel:
        tlabel = slabel[5:10] + [s4,orb4,x4,y4,z4]
        tmp24 = reorder_z(tlabel)
        if tmp24 == tlabel:
            slabel2 = slabel[0:10]+ [s4,orb4,x4,y4,z4] + slabel[10:15]
        elif  tmp24 != tlabel:
            tlabel = slabel[0:5] + [s4,orb4,x4,y4,z4]   
            tmp14 = reorder_z(tlabel)
            if tmp14 == tlabel:
                slabel2 = slabel[0:5]+ [s4,orb4,x4,y4,z4] + slabel[5:15]
            elif  tmp14 != tlabel:
                slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15]     
                
    tlabel = slabel2[15:20] + [s5,orb5,x5,y5,z5]
    tmp45 = reorder_z(tlabel)                
    if tmp45 == tlabel:
        slabel3 = slabel2 + [s5,orb5,x5,y5,z5]
    else:
        tlabel = slabel2[10:15] + [s5,orb5,x5,y5,z5] 
        tmp35 = reorder_z(tlabel)                         
        if tmp35 == tlabel:
            slabel3 = slabel2[0:15] + [s5,orb5,x5,y5,z5] + slabel2[15:20]
        else:
            tlabel = slabel2[5:10] + [s5,orb5,x5,y5,z5] 
            tmp25 = reorder_z(tlabel)                          
            if tmp25 == tlabel:
                slabel3 = slabel2[0:10] + [s5,orb5,x5,y5,z5] + slabel2[10:20]   
            else:
                tlabel = slabel2[0:5] + [s5,orb5,x5,y5,z5] 
                tmp15 = reorder_z(tlabel)                          
                if tmp15 == tlabel:
                    slabel3 = slabel2[0:5] + [s5,orb5,x5,y5,z5] + slabel2[5:20]     
                else:
                    slabel3 = [s5,orb5,x5,y5,z5] + slabel2                       
                        
                
                
    return slabel3


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val,bonding_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
# #     for ii in range(0,1325):
# #         for jj in range(0,1325):
# #             if M_dense[ii,jj]>0 and ii!=jj:
# #                 print (ii,jj,M_dense[ii,jj])
# #             if M_dense[ii,jj]==0 and ii==jj:
# #                 print (ii,jj,M_dense[ii,jj])
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=20, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    print (vals[0])
    
    if abs(vals[0]-vals[3])<10**(-5):
        number = 4
    elif abs(vals[0]-vals[2])<10**(-5):
        number = 3        
    elif abs(vals[0]-vals[1])<10**(-5):
        number = 2
    else:
        number = 1
    print ('Degeneracy of ground state is ' ,number)      
    
    wgt_LmLn = np.zeros(40)
    wgt_d8d8L = np.zeros(40)
    wgt_d8Ld8 = np.zeros(40)
    wgt_d9Ld8L = np.zeros(40) 
    wgt_d8Ld9L = np.zeros(40)         
    wgt_d9L2d8= np.zeros(40)
    wgt_d8d9L2= np.zeros(40)        
    wgt_d9d8L2= np.zeros(40)
    wgt_d9L3d9= np.zeros(40)   
    wgt_d9L2d9L= np.zeros(40)  
    wgt_d9Ld9L2= np.zeros(40)
    wgt_d9d9L3= np.zeros(40)        
    wgt_d8L3d10= np.zeros(40)   
    wgt_d8L2d10L= np.zeros(40)  
    wgt_d8Ld10L2= np.zeros(40)
    wgt_d8d10L3= np.zeros(40) 
    wgt_d9d10L4= np.zeros(40)
    wgt_d9Ld10L3= np.zeros(40)        
    wgt_d9L2d10L2= np.zeros(40)          
    wgt_d9L3d10L= np.zeros(40)   
    wgt_d9L4d10= np.zeros(40)  
    wgt_a1= np.zeros(10)   
    wgt_b1= np.zeros(10)       
    wgt_L= np.zeros(10) 
    wgt_O= np.zeros(10)      

    wgt_s= np.zeros(40)         
    wgt_ds= np.zeros(40)           
    wgt_dds= np.zeros(80)    
    wgt_ddds= np.zeros(80)           
    wgt_dddds= np.zeros(80)         
    sumweight=0
    sumweight1=0
    synweight2=0
    
    test= np.zeros(10)  
    
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,number):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        print ('k = ', k)        
        indices = np.nonzero(abs(vecs[:,k])>0.1)

       
        
#         s11=0
#         s10=0        
#         s01=0
#         s00=0        
        #Sumweight refers to the general weight.Sumweight1 refers to the weight in indices.Sumweight_picture refers to the weight that is calculated.Sumweight2 refers to the weight that differs by orbits


        # stores all weights for sorting later
        dim = len(vecs[:,k])
        allwgts = np.zeros(dim)
        allwgts = abs(vecs[:,k])**2
        ilead = np.argsort(-allwgts)   # argsort returns small value first by default
            

        total = 0

        print ("Compute the weights in GS (lowest Aw peak)")
        
        #for i in indices[0]:
        for i in range(dim):
            # state is original state but its orbital info remains after basis change
            istate = ilead[i]
            weight = allwgts[istate]
            
            #if weight>0.01:

            total += weight
                
            state = VS.get_state(VS.lookup_tbl[istate])
            
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin'] 
            s5 = state['hole5_spin']             
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb'] 
            orb5 = state['hole5_orb']             
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']  
            x5, y5, z5 = state['hole5_coord']              

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Ni_12  = S_Ni_val[istate]
            Sz_Ni_12 = Sz_Ni_val[istate]
            S_Cu_12  = S_Cu_val[istate]
            Sz_Cu_12 = Sz_Cu_val[istate]
            bonding = bonding_val[istate]
#             S_Niother_12  = S_other_Ni_val[i]
#             Sz_Niother_12 = Sz_other_Ni_val[i]
#             S_Cuother_12  = S_other_Cu_val[i]
#             Sz_Cuother_12 = Sz_other_Cu_val[i]

#             print ( i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,' ',orb5,s5,x5,y5,z5,\
#                '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
#                ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
#                ", weight = ", weight,'\n')                     

            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,s5,orb5,x5,y5,z5]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
            s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];     
            s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];                
    
    
      
    
    
            if weight >0.001:
                sumweight1=sumweight1+abs(vecs[istate,k])**2
                print (' state ', istate, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,' ',orb5,s5,x5,y5,z5,\
                   '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                   ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, ',  bonding=',bonding, \
                   ", weight = ", weight,'\n')  
                
                
            if bonding==0:
                test[6]+=abs(vecs[istate,k])**2 
            if bonding==1:
                test[7]+=abs(vecs[istate,k])**2 
            if bonding==-1:
                test[8]+=abs(vecs[istate,k])**2                 
                
                
            if orb1 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb1 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb1 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb1 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2       
                
            if orb2 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb2 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb2 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb2 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2                       
                
            if orb3 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb3 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb3 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb3 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2       
                
            if orb4 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb4 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb4 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb4 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2  
                
            if orb5 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb5 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb5 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb5 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2                       
                
            if (orb1 in pam.O_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs)  and  (orb4 in pam.O_orbs) and  (orb5 in pam.O_orbs): 
                wgt_LmLn[0]+=abs(vecs[istate,k])**2 
                if bonding==0:
                    wgt_LmLn[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_LmLn[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_LmLn[38]+=abs(vecs[istate,k])**2                     

        

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==z5==2 and z3==z4==0 : 
                wgt_d8Ld8[0]+=abs(vecs[istate,k])**2
                if orb1==orb2==orb3==orb4=='dx2y2':
                    wgt_d8Ld8[1]+=abs(vecs[istate,k])**2                
                elif orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
                    wgt_d8Ld8[2]+=abs(vecs[istate,k])**2  
                elif orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
                    wgt_d8Ld8[3]+=abs(vecs[istate,k])**2                      
                elif orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
                    wgt_d8Ld8[4]+=abs(vecs[istate,k])**2                        
                if bonding==0:
                    wgt_d8Ld8[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8Ld8[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8Ld8[38]+=abs(vecs[istate,k])**2   
                
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
                wgt_d8d8L[0]+=abs(vecs[istate,k])**2             
                if orb1==orb2==orb3==orb4=='dx2y2':
                    wgt_d8d8L[1]+=abs(vecs[istate,k])**2                
                elif orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
                    wgt_d8d8L[2]+=abs(vecs[istate,k])**2                
                elif orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
                    wgt_d8d8L[3]+=abs(vecs[istate,k])**2                      
                elif orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
                    wgt_d8d8L[4]+=abs(vecs[istate,k])**2
                if bonding==0:
                    wgt_d8d8L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d8L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d8L[38]+=abs(vecs[istate,k])**2                                             

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==2 and z2==z3==z5==0 : 
                wgt_d9Ld8L[0]+=abs(vecs[istate,k])**2
                if orb1==orb2==orb3=='dx2y2':
                    wgt_d9Ld8L[1]+=abs(vecs[istate,k])**2
                elif orb1==orb3=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9Ld8L[2]+=abs(vecs[istate,k])**2                    
                    if S_Cu_12 ==0 and S_Ni_12 ==0: 
                        wgt_d9Ld8L[3]+=abs(vecs[istate,k])**2 
                    elif S_Cu_12 ==1: 
                        wgt_d9Ld8L[4]+=abs(vecs[istate,k])**2   
                elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
                    wgt_d9Ld8L[5]+=abs(vecs[istate,k])**2                          
                if bonding==0:
                    wgt_d9Ld8L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld8L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld8L[38]+=abs(vecs[istate,k])**2                     
                    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z4==2 and z3==z5==0 : 
                wgt_d8Ld9L[0]+=abs(vecs[istate,k])**2 
                if orb1==orb2==orb3=='dx2y2':
                    wgt_d8Ld9L[1]+=abs(vecs[istate,k])**2
                elif orb2==orb3=='dx2y2' and orb1=='d3z2r2':
                    wgt_d8Ld9L[2]+=abs(vecs[istate,k])**2 
                    if S_Ni_12 ==0 and S_Cu_12 ==0: 
                        wgt_d8Ld9L[3]+=abs(vecs[istate,k])**2 
             
                    elif S_Ni_12 ==1: 
                        wgt_d8Ld9L[4]+=abs(vecs[istate,k])**2    
                elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
                    wgt_d8Ld9L[5]+=abs(vecs[istate,k])**2  
                    
                if bonding==0:
                    wgt_d8Ld9L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8Ld9L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8Ld9L[38]+=abs(vecs[istate,k])**2                     
                    
            
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==z5==2 and z2==z3==0 : 
                wgt_d9L2d8[0]+=abs(vecs[istate,k])**2 
                if orb1==orb2==orb3=='dx2y2':
                    wgt_d9L2d8[1]+=abs(vecs[istate,k])**2
                elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
                    wgt_d9L2d8[2]+=abs(vecs[istate,k])**2                  
                elif orb1==orb3=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9L2d8[3]+=abs(vecs[istate,k])**2   
                    
                if bonding==0:
                    wgt_d9L2d8[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L2d8[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L2d8[38]+=abs(vecs[istate,k])**2                      
                    
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
                wgt_d8d9L2[0]+=abs(vecs[istate,k])**2   
                if orb1==orb2==orb3=='dx2y2':
                    wgt_d8d9L2[1]+=abs(vecs[istate,k])**2
                elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
                    wgt_d8d9L2[2]+=abs(vecs[istate,k])**2                 
                elif orb2==orb3=='dx2y2' and orb1=='d3z2r2':
                    wgt_d8d9L2[3]+=abs(vecs[istate,k])**2    
                    
                if bonding==0:
                    wgt_d8d9L2[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d9L2[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d9L2[38]+=abs(vecs[istate,k])**2                      
                    
                    
                    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
                wgt_d9d8L2[0]+=abs(vecs[istate,k])**2  
                
                if bonding==0:
                    wgt_d9d8L2[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d8L2[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d8L2[38]+=abs(vecs[istate,k])**2                  
                
                
                

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
                wgt_d9d9L3[0]+=abs(vecs[istate,k])**2  
                
                if bonding==0:
                    wgt_d9d9L3[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d9L3[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d9L3[38]+=abs(vecs[istate,k])**2                  
                
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==2 and z2==z4==z5==0 : 
                wgt_d9Ld9L2[0]+=abs(vecs[istate,k])**2  
                if orb1==orb2=='dx2y2':
                    wgt_d9Ld9L2[1]+=abs(vecs[istate,k])**2
                elif orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9Ld9L2[2]+=abs(vecs[istate,k])**2
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9Ld9L2[3]+=abs(vecs[istate,k])**2                    
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9Ld9L2[4]+=abs(vecs[istate,k])**2        
                    
                if bonding==0:
                    wgt_d9Ld9L2[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld9L2[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld9L2[38]+=abs(vecs[istate,k])**2                      
                    
                    
                    
                    
                    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==2 and z2==z5==0 : 
                wgt_d9L2d9L[0]+=abs(vecs[istate,k])**2    
                if orb1==orb2=='dx2y2':
                    wgt_d9L2d9L[1]+=abs(vecs[istate,k])**2
                elif orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9L2d9L[2]+=abs(vecs[istate,k])**2
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9L2d9L[3]+=abs(vecs[istate,k])**2                    
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9L2d9L[4]+=abs(vecs[istate,k])**2    
                    
                if bonding==0:
                    wgt_d9L2d9L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L2d9L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L2d9L[38]+=abs(vecs[istate,k])**2                      
                    
                    
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==z5==2 and z2==0 : 
                wgt_d9L3d9[0]+=abs(vecs[istate,k])**2         
                
                if bonding==0:
                    wgt_d9L3d9[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L3d9[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L3d9[38]+=abs(vecs[istate,k])**2                  
                
                
                

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
                wgt_d8d10L3[0]+=abs(vecs[istate,k])**2  
                
                if bonding==0:
                    wgt_d8d10L3[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d10L3[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d10L3[38]+=abs(vecs[istate,k])**2                  
                
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==2 and z4==z5==0 : 
                wgt_d8Ld10L2[0]+=abs(vecs[istate,k])**2  
                
                if bonding==0:
                    wgt_d8Ld10L2[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8Ld10L2[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8Ld10L2[38]+=abs(vecs[istate,k])**2                  
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==2 and z5==0 : 
                wgt_d8L2d10L[0]+=abs(vecs[istate,k])**2    
                
                if bonding==0:
                    wgt_d8L2d10L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8L2d10L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8L2d10L[38]+=abs(vecs[istate,k])**2                  
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==2 : 
                wgt_d8L3d10[0]+=abs(vecs[istate,k])**2                  
                
                if bonding==0:
                    wgt_d8L3d10[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8L3d10[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8L3d10[38]+=abs(vecs[istate,k])**2                  
                
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
                wgt_d9d10L4[0]+=abs(vecs[istate,k])**2           
                
                if bonding==0:
                    wgt_d9d10L4[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d10L4[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d10L4[38]+=abs(vecs[istate,k])**2                  
                
                
                
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
                wgt_d9Ld10L3[0]+=abs(vecs[istate,k])**2      
                
                if bonding==0:
                    wgt_d9Ld10L3[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld10L3[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld10L3[38]+=abs(vecs[istate,k])**2                  
                
                
                
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==2 and z4==z5==0 : 
                wgt_d9L2d10L2[0]+=abs(vecs[istate,k])**2     
                
                if bonding==0:
                    wgt_d9L2d10L2[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L2d10L2[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L2d10L2[38]+=abs(vecs[istate,k])**2                  
                
                
                
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==2 and z5==0 : 
                wgt_d9L3d10L[0]+=abs(vecs[istate,k])**2    
                
                if bonding==0:
                    wgt_d9L3d10L[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L3d10L[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L3d10L[38]+=abs(vecs[istate,k])**2                  
                
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==2: 
                wgt_d9L4d10[0]+=abs(vecs[istate,k])**2 
                
                if bonding==0:
                    wgt_d9L4d10[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L4d10[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L4d10[38]+=abs(vecs[istate,k])**2                     
                
                
            elif (orb1 in pam.Obilayer_orbs): 
                wgt_s[0]+=abs(vecs[istate,k])**2  
                
                if bonding==0:
                    wgt_s[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_s[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_s[38]+=abs(vecs[istate,k])**2                    
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Obilayer_orbs): 
                wgt_ds[0]+=abs(vecs[istate,k])**2   
                if orb3 in pam.Obilayer_orbs:
                    wgt_ds[1]+=abs(vecs[istate,k])**2 
                    if z1==2 and z4==2 and z5==2:
                        wgt_ds[3]+=abs(vecs[istate,k])**2*2 
                        if orb1=='dx2y2':
                            wgt_ds[9]+=abs(vecs[istate,k])**2*2  
                    if z1==2 and z4==2 and z5==0:
                        wgt_ds[4]+=abs(vecs[istate,k])**2*2 
                        if orb1=='dx2y2':
                            wgt_ds[10]+=abs(vecs[istate,k])**2 *2                         
                    if z1==2 and z4==0 and z5==0:
                        wgt_ds[5]+=abs(vecs[istate,k])**2*2                              
                        if orb1=='dx2y2':
                            wgt_ds[11]+=abs(vecs[istate,k])**2*2                          
                    
                elif orb3 in pam.O_orbs:
                    wgt_ds[2]+=abs(vecs[istate,k])**2  
                    if z1==2 and z3==2 and z4==2 and z5==0:
                        wgt_ds[6]+=abs(vecs[istate,k])**2*2  
                        if orb1=='dx2y2':
                            wgt_ds[12]+=abs(vecs[istate,k])**2*2                               
                    if z1==2 and z3==2 and z4==0 and z5==0:
                        wgt_ds[7]+=abs(vecs[istate,k])**2*2   
                        if orb1=='dx2y2':
                            wgt_ds[13]+=abs(vecs[istate,k])**2*2                               
                    if z1==2 and z3==0 and z4==0 and z5==0:
                        wgt_ds[8]+=abs(vecs[istate,k])**2*2                         
                        if orb1=='dx2y2':
                            wgt_ds[14]+=abs(vecs[istate,k])**2 *2    
                            
                if bonding==0:
                    wgt_ds[36]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_ds[37]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_ds[38]+=abs(vecs[istate,k])**2                                
                            
                            
                
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Obilayer_orbs): 
                wgt_dds[0]+=abs(vecs[istate,k])**2       
                if (orb4 in pam.Obilayer_orbs) and (orb5 in pam.O_orbs):
                    wgt_dds[1]+=abs(vecs[istate,k])**2
                    if z1==z2==2:
                        wgt_dds[3]+=abs(vecs[istate,k])**2
                        if z4==z5==2:
                            wgt_dds[57]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[9]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[10]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[11]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[12]+=abs(vecs[istate,k])**2                              
                        elif z4==2 and z5==0:
                            wgt_dds[58]+=abs(vecs[istate,k])**2                           
                            if orb1==orb2=='dx2y2':
                                wgt_dds[33]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[34]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[35]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[36]+=abs(vecs[istate,k])**2       
                                
                    if z1==2 and z2==0:
                        wgt_dds[4]+=abs(vecs[istate,k])**2 
                        if z4==z5==2:
                            wgt_dds[59]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[13]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[14]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[15]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[16]+=abs(vecs[istate,k])**2                                
                        elif z4==2 and z5==0:
                            wgt_dds[60]+=abs(vecs[istate,k])**2                           
                            if orb1==orb2=='dx2y2':
                                wgt_dds[37]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[38]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[39]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[40]+=abs(vecs[istate,k])**2                          
                    if z1==0 and z2==0:
                        wgt_dds[5]+=abs(vecs[istate,k])**2   
                        if z4==z5==2:
                            wgt_dds[61]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[17]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[18]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[19]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[20]+=abs(vecs[istate,k])**2                                  
                        elif z4==2 and z5==0:
                            wgt_dds[62]+=abs(vecs[istate,k])**2                           
                            if orb1==orb2=='dx2y2':
                                wgt_dds[41]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[42]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[43]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[44]+=abs(vecs[istate,k])**2   
                                
                    if bonding==0:
                        wgt_dds[73]+=abs(vecs[istate,k])**2 
                    if bonding==1:
                        wgt_dds[74]+=abs(vecs[istate,k])**2 
                    if bonding==-1:
                        wgt_dds[75]+=abs(vecs[istate,k])**2                                    
                            
                if (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs):
                    wgt_dds[2]+=abs(vecs[istate,k])**2        
                    if z1==z2==2:
                        wgt_dds[6]+=abs(vecs[istate,k])**2
                        if z4==z5==2:
                            wgt_dds[63]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[21]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[22]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[23]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[24]+=abs(vecs[istate,k])**2                              
                        elif z4==2 and z5==0:
                            wgt_dds[64]+=abs(vecs[istate,k])**2                            
                            if orb1==orb2=='dx2y2':
                                wgt_dds[45]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[46]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[47]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[48]+=abs(vecs[istate,k])**2  
                    if z1==2 and z2==0:
                        wgt_dds[7]+=abs(vecs[istate,k])**2 
                        if z4==z5==2:
                            wgt_dds[65]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[25]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[26]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[27]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[28]+=abs(vecs[istate,k])**2                                               
                        elif z4==2 and z5==0:
                            wgt_dds[66]+=abs(vecs[istate,k])**2                           
                            if orb1==orb2=='dx2y2':
                                wgt_dds[49]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[50]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[51]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[52]+=abs(vecs[istate,k])**2      
                                
                    if z1==0 and z2==0:
                        wgt_dds[8]+=abs(vecs[istate,k])**2  
                        if z4==z5==2:
                            wgt_dds[67]+=abs(vecs[istate,k])**2
                            if orb1==orb2=='dx2y2':
                                wgt_dds[29]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[30]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[31]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[32]+=abs(vecs[istate,k])**2                                 
                        elif z4==2 and z5==0:
                            wgt_dds[68]+=abs(vecs[istate,k])**2                           
                            if orb1==orb2=='dx2y2':
                                wgt_dds[53]+=abs(vecs[istate,k])**2
                            elif orb1=='d3z2r2' and orb2=='dx2y2':
                                wgt_dds[54]+=abs(vecs[istate,k])**2
                            elif orb1=='dx2y2' and orb2=='d3z2r2':
                                wgt_dds[55]+=abs(vecs[istate,k])**2                    
                            elif orb1=='d3z2r2' and orb2=='d3z2r2':
                                wgt_dds[56]+=abs(vecs[istate,k])**2                       
                    
                    if bonding==0:
                        wgt_dds[76]+=abs(vecs[istate,k])**2 
                    if bonding==1:
                        wgt_dds[77]+=abs(vecs[istate,k])**2 
                    if bonding==-1:
                        wgt_dds[78]+=abs(vecs[istate,k])**2    
                    
                    
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Obilayer_orbs): 
                wgt_ddds[0]+=abs(vecs[istate,k])**2  
                if orb5 in pam.Obilayer_orbs:
                    wgt_ddds[1]+=abs(vecs[istate,k])**2 
                    if z1==z2==2 and z3==0:
                        wgt_ddds[3]+=abs(vecs[istate,k])**2
                        if orb1==orb2==orb3=='dx2y2':
                            wgt_ddds[7]+=abs(vecs[istate,k])**2 
                        elif orb1=='d3z2r2' and orb2==orb3=='dx2y2':
                            wgt_ddds[8]+=abs(vecs[istate,k])**2                             
                        elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
                            wgt_ddds[9]+=abs(vecs[istate,k])**2  
                           
                            
                            
                    if z1==2 and z2==z3==0:
                        wgt_ddds[4]+=abs(vecs[istate,k])**2                                                  
                        if orb1==orb2==orb3=='dx2y2':
                            wgt_ddds[10]+=abs(vecs[istate,k])**2 
                        elif orb2=='d3z2r2' and orb1==orb3=='dx2y2':
                            wgt_ddds[11]+=abs(vecs[istate,k])**2                             
                        elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
                            wgt_ddds[12]+=abs(vecs[istate,k])**2 
                            
                    if bonding==0:
                        wgt_ddds[73]+=abs(vecs[istate,k])**2 
                    if bonding==1:
                        wgt_ddds[74]+=abs(vecs[istate,k])**2 
                    if bonding==-1:
                        wgt_ddds[75]+=abs(vecs[istate,k])**2                                    
                            
                if orb5 in pam.O_orbs:
                    wgt_ddds[2]+=abs(vecs[istate,k])**2                 
                    if z1==z2==2 and z3==0:
                        wgt_ddds[5]+=abs(vecs[istate,k])**2  
                        if orb1==orb2==orb3=='dx2y2':
                            wgt_ddds[13]+=abs(vecs[istate,k])**2 
                        elif orb1=='d3z2r2' and orb2==orb3=='dx2y2':
                            wgt_ddds[14]+=abs(vecs[istate,k])**2                             
                        elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
                            wgt_ddds[15]+=abs(vecs[istate,k])**2  
                        if z5==2:
                            wgt_ddds[19]+=abs(vecs[istate,k])**2*2  
                        if z5==0:
                            wgt_ddds[20]+=abs(vecs[istate,k])**2*2      
                            
                            
                            
                        
                    if z1==2 and z2==z3==0:
                        wgt_ddds[6]+=abs(vecs[istate,k])**2                       
                        if orb1==orb2==orb3=='dx2y2':
                            wgt_ddds[16]+=abs(vecs[istate,k])**2 
                        elif orb2=='d3z2r2' and orb1==orb3=='dx2y2':
                            wgt_ddds[17]+=abs(vecs[istate,k])**2                             
                        elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
                            wgt_ddds[18]+=abs(vecs[istate,k])**2         
                            
                            
                    if bonding==0:
                        wgt_ddds[76]+=abs(vecs[istate,k])**2 
                    if bonding==1:
                        wgt_ddds[77]+=abs(vecs[istate,k])**2 
                    if bonding==-1:
                        wgt_ddds[78]+=abs(vecs[istate,k])**2                             
                            
                            
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.Obilayer_orbs): 
                wgt_dddds[0]+=abs(vecs[istate,k])**2                  
                if orb1==orb2==orb3==orb4=='dx2y2':
                    wgt_dddds[1]+=abs(vecs[istate,k])**2  
                if orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
                    wgt_dddds[2]+=abs(vecs[istate,k])**2                  
                if orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
                    wgt_dddds[3]+=abs(vecs[istate,k])**2   
                if orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
                    wgt_dddds[4]+=abs(vecs[istate,k])**2                     
                if orb1==orb2==orb3==orb4=='d3z2r2':
                    wgt_dddds[5]+=abs(vecs[istate,k])**2  
                    
                if bonding==0:
                    wgt_dddds[76]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_dddds[77]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_dddds[78]+=abs(vecs[istate,k])**2                     
                    

            sumweight=sumweight+abs(vecs[istate,k])**2

    print ('sumweight=',sumweight/number)
    print ('LmLn=',wgt_LmLn[0]/number,'  bonding0=',wgt_LmLn[36]/number,'  bonding1=',wgt_LmLn[37]/number,'  bonding-1=',wgt_LmLn[38]/number)
    print ('d8d8L=',wgt_d8d8L[0]/number+wgt_d8Ld8[0]/number,'  bonding0=',wgt_d8d8L[36]/number+wgt_d8Ld8[36]/number,'  bonding1=',wgt_d8d8L[37]/number+wgt_d8Ld8[37]/number,'  bonding-1=',wgt_d8d8L[38]/number+wgt_d8Ld8[38]/number)
    print ('d9Ld8L=',wgt_d9Ld8L[0]/number+wgt_d8Ld9L[0]/number,'  bonding0=',wgt_d9Ld8L[36]/number+wgt_d8Ld9L[36]/number,'  bonding1=',wgt_d9Ld8L[37]/number+wgt_d8Ld9L[37]/number,'  bonding-1=',wgt_d9Ld8L[38]/number+wgt_d8Ld9L[38]/number) 
      
#     print ('d9d8L2=',wgt_d9d8L2[0]/number,'  bonding0=',wgt_d9d8L2[36]/number,'  bonding1=',wgt_d9d8L2[37]/number,'  bonding-1=',wgt_d9d8L2[38]/number)          
    print ('d9L2d8=',wgt_d9L2d8[0]/number+wgt_d8d9L2[0]/number,'  bonding0=',wgt_d9L2d8[36]/number+wgt_d8d9L2[36]/number,'  bonding1=',wgt_d9L2d8[37]/number+wgt_d8d9L2[37]/number,'  bonding-1=',wgt_d9L2d8[38]/number+wgt_d8d9L2[38]/number) 
       
    print ('d9L3d9=',wgt_d9L3d9[0]/number+wgt_d9d9L3[0]/number,'  bonding0=',wgt_d9L3d9[36]/number+wgt_d9d9L3[36]/number,'  bonding1=',wgt_d9L3d9[37]/number+wgt_d9d9L3[37]/number,'  bonding-1=',wgt_d9L3d9[38]/number+wgt_d9d9L3[38]/number) 
    print ('d9L2d9L=',wgt_d9L2d9L[0]/number+wgt_d9Ld9L2[0]/number,'  bonding0=',wgt_d9L2d9L[36]/number+wgt_d9Ld9L2[36]/number,'  bonding1=',wgt_d9L2d9L[37]/number+wgt_d9Ld9L2[37]/number,'  bonding-1=',wgt_d9L2d9L[38]/number+wgt_d9Ld9L2[38]/number)         

#     print ('d8L3d10=',wgt_d8L3d10[0]/number,'  bonding0=',wgt_d8L3d10[36]/number,'  bonding1=',wgt_d8L3d10[37]/number,'  bonding-1=',wgt_d8L3d10[38]/number) 
#     print ('d8L2d10L=',wgt_d8L2d10L[0]/number,'  bonding0=',wgt_d8L2d10L[36]/number,'  bonding1=',wgt_d8L2d10L[37]/number,'  bonding-1=',wgt_d8L2d10L[38]/number)         
#     print ('d8Ld10L2=',wgt_d8Ld10L2[0]/number,'  bonding0=',wgt_d8Ld10L2[36]/number,'  bonding1=',wgt_d8Ld10L2[37]/number,'  bonding-1=',wgt_d8Ld10L2[38]/number)         
#     print ('d8d10L3=',wgt_d8d10L3[0]/number,'  bonding0=',wgt_d8d10L3[36]/number,'  bonding1=',wgt_d8d10L3[37]/number,'  bonding-1=',wgt_d8d10L3[38]/number)  
#     print ('d9d10L4=',wgt_d9d10L4[0]/number,'  bonding0=',wgt_d9d10L4[36]/number,'  bonding1=',wgt_d9d10L4[37]/number,'  bonding-1=',wgt_d9d10L4[38]/number)          
#     print ('d9Ld10L3=',wgt_d9Ld10L3[0]/number,'  bonding0=',wgt_d9Ld10L3[36]/number,'  bonding1=',wgt_d9Ld10L3[37]/number,'  bonding-1=',wgt_d9Ld10L3[38]/number)         
#     print ('d9L2d10L2=',wgt_d9L2d10L2[0]/number,'  bonding0=',wgt_d9L2d10L2[36]/number,'  bonding1=',wgt_d9L2d10L2[37]/number,'  bonding-1=',wgt_d9L2d10L2[38]/number)         
#     print ('d9L3d10L=',wgt_d9L3d10L[0]/number,'  bonding0=',wgt_d9L3d10L[36]/number,'  bonding1=',wgt_d9L3d10L[37]/number,'  bonding-1=',wgt_d9L3d10L[38]/number) 
#     print ('d9L4d10=',wgt_d9L4d10[0]/number,'  bonding0=',wgt_d9L4d10[36]/number,'  bonding1=',wgt_d9L4d10[37]/number,'  bonding-1=',wgt_d9L4d10[38]/number) 
    print ('s=',wgt_s[0]/number,'  bonding0=',wgt_s[36]/number,'  bonding1=',wgt_s[37]/number,'  bonding-1=',wgt_s[38]/number) 
    
    print ('d9-O2-L2 + d9L-O2-L + d9L2-O2=',wgt_ds[1]/number,'  bonding0=',wgt_ds[33]/number,'  bonding1=',wgt_ds[34]/number,'  bonding-1=',wgt_ds[35]/number)        
    print ('d9-O-L3 + d9L-O-L2 + d9L2-O-L + d9L3-O=',wgt_ds[2]/number,'  bonding0=',wgt_ds[36]/number,'  bonding1=',wgt_ds[37]/number,'  bonding-1=',wgt_ds[38]/number)             
    print ('d8-O2-L + d8L-O2 + d9-O2-L2 + d9L-O2-L + d9L2-O2=',wgt_dds[1]/number,'  bonding0=',wgt_dds[73]/number,'  bonding1=',wgt_dds[74]/number,'  bonding-1=',wgt_dds[75]/number)  
    print ('d8-O-L2 + d8L-O-L + d8L2-O + d9-O-d9L2 + d9L-O-d9L=',wgt_dds[2]/number,'  bonding0=',wgt_dds[76]/number,'  bonding1=',wgt_dds[77]/number,'  bonding-1=',wgt_dds[78]/number)  
    print ('d8-O2-d9=',wgt_ddds[1]/number,'  bonding0=',wgt_ddds[73]/number,'  bonding1=',wgt_ddds[74]/number,'  bonding-1=',wgt_ddds[75]/number) 
    print ('d8L-O-d9 + d8-O-d9L=',wgt_ddds[2]/number,'  bonding0=',wgt_ddds[76]/number,'  bonding1=',wgt_ddds[77]/number,'  bonding-1=',wgt_ddds[78]/number)  
    print ('d8-O-d8=',wgt_dddds[0]/number,'  bonding0=',wgt_dddds[76]/number,'  bonding1=',wgt_dddds[77]/number,'  bonding-1=',wgt_dddds[78]/number)   

    print ('test[0]=',test[0]/number)     
    print ('test[1]=',test[1]/number)        
    print ('test[2]=',test[2]/number)        
    print ('test[3]=',test[3]/number)  
    print ('test[4]=',test[4]/number)        
    print ('test[5]=',test[5]/number)      
    print ('test[6]=',test[6]/number)  
    print ('test[7]=',test[7]/number)        
    print ('test[8]=',test[8]/number)          
    
    
    sumweight2 = wgt_LmLn[0]+wgt_d8Ld8[0]+wgt_d8d8L[0]+wgt_d9Ld8L[0]+wgt_d8Ld9L[0]+wgt_d9d8L2[0]+wgt_d9L2d8[0]+wgt_d8d9L2[0]\
             +wgt_d9L3d9[0]+wgt_d9L2d9L[0]+wgt_d9Ld9L2[0]+wgt_d9d9L3[0]+wgt_d8L3d10[0]+wgt_d8L2d10L[0]\
             +wgt_d8Ld10L2[0]+wgt_d8d10L3[0]+wgt_d9d10L4[0]+wgt_d9Ld10L3[0]+wgt_d9L2d10L2[0]+wgt_d9L3d10L[0]+wgt_d9L4d10[0]
    print ('sumweight2=',sumweight2/number)
#         print ('s11=',s11)        
#         print ('s10=',s10)       
#         print ('s01=',s01)  
#         print ('s00=',s00)          







    path = './data'		# create file

    if os.path.isdir(path) == False:
        os.mkdir(path) 

    txt=open('./data/value','a')                                  
    txt.write(str(vals[0])+'\n')
    txt.close()
        
    txt=open('./data/a1','a')                                  
    txt.write(str(wgt_a1[0]/number)+'\n')
    txt.close()           
    txt=open('./data/b1','a')                                  
    txt.write(str(wgt_b1[0]/number)+'\n')
    txt.close()     
    txt=open('./data/L','a')                                  
    txt.write(str(wgt_L[0]/number)+'\n')
    txt.close()         
    txt=open('./data/O','a')                                  
    txt.write(str(wgt_O[0]/number)+'\n')
    txt.close()         
    
    

    txt=open('./data/LmLn','a')                                  
    txt.write(str(wgt_LmLn[0]/number)+'\n')
    txt.close()     
    txt=open('./data/LmLn_b0','a')                                  
    txt.write(str(wgt_LmLn[36]/number)+'\n')
    txt.close()   
    txt=open('./data/LmLn_b1','a')                                  
    txt.write(str(wgt_LmLn[37]/number)+'\n')
    txt.close()       
    txt=open('./data/LmLn_b-1','a')                                  
    txt.write(str(wgt_LmLn[38]/number)+'\n')
    txt.close()       
    
    txt=open('./data/d8Ld8','a')                                  
    txt.write(str(wgt_d8Ld8[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d8Ld8_b1b1b1b1','a')                                  
    txt.write(str(wgt_d8Ld8[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d8Ld8_a1b1a1b1','a')                                  
    txt.write(str(wgt_d8Ld8[2]/number)+'\n')
    txt.close()     
    txt=open('./data/d8Ld8_a1b1b1b1','a')                                  
    txt.write(str(wgt_d8Ld8[3]/number)+'\n')
    txt.close()             
    txt=open('./data/d8Ld8_b1b1a1b1','a')                                  
    txt.write(str(wgt_d8Ld8[4]/number)+'\n')
    txt.close()   
    txt=open('./data/d8Ld8+d8d8L','a')                                  
    txt.write(str(wgt_d8Ld8[0]/number+wgt_d8d8L[0]/number)+'\n')
    txt.close()   
    txt=open('./data/d8Ld8_b0','a')                                  
    txt.write(str(wgt_d8Ld8[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8Ld8_b1','a')                                  
    txt.write(str(wgt_d8Ld8[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8Ld8_b-1','a')                                  
    txt.write(str(wgt_d8Ld8[38]/number)+'\n')
    txt.close()           
    
    
    

    txt=open('./data/d8d8L','a')                                  
    txt.write(str(wgt_d8d8L[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d8L_b1b1b1b1','a')                                  
    txt.write(str(wgt_d8d8L[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d8L_a1b1a1b1','a')                                  
    txt.write(str(wgt_d8d8L[2]/number)+'\n')
    txt.close()              
    txt=open('./data/d8d8L_a1b1b1b1','a')                                  
    txt.write(str(wgt_d8d8L[3]/number)+'\n')
    txt.close()             
    txt=open('./data/d8d8L_b1b1a1b1','a')                                  
    txt.write(str(wgt_d8d8L[4]/number)+'\n')
    txt.close() 
    txt=open('./data/d8d8L_b0','a')                                  
    txt.write(str(wgt_d8d8L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8d8L_b1','a')                                  
    txt.write(str(wgt_d8d8L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8d8L_b-1','a')                                  
    txt.write(str(wgt_d8d8L[38]/number)+'\n')
    txt.close()         



    txt=open('./data/d9Ld8L','a')                                  
    txt.write(str(wgt_d9Ld8L[0]/number)+'\n')
    txt.close()         
    txt=open('./data/d9Ld8L_b1b1b1','a')                                  
    txt.write(str(wgt_d9Ld8L[1]/number)+'\n')
    txt.close()        
    txt=open('./data/d9Ld8L_b1a1b1','a')                                  
    txt.write(str(wgt_d9Ld8L[2]/number)+'\n')
    txt.close()     
    txt=open('./data/d9Ld8L_b1a1b1_0','a')                                  
    txt.write(str(wgt_d9Ld8L[3]/number)+'\n')
    txt.close()            
    txt=open('./data/d9Ld8L_b1a1b1_1','a')                                  
    txt.write(str(wgt_d9Ld8L[4]/number)+'\n')
    txt.close()        
    txt=open('./data/d9Ld8L_a1a1b1','a')                                  
    txt.write(str(wgt_d9Ld8L[5]/number)+'\n')
    txt.close() 
    txt=open('./data/d9Ld8L+d8Ld9L','a')                                  
    txt.write(str(wgt_d9Ld8L[0]/number+wgt_d8Ld9L[0]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld8L_b1a1b1_d8Ld9L','a')                                  
    txt.write(str(wgt_d9Ld8L[2]*2/number)+'\n')
    txt.close()     
    txt=open('./data/d9Ld8L_b0','a')                                  
    txt.write(str(wgt_d9Ld8L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9Ld8L_b1','a')                                  
    txt.write(str(wgt_d9Ld8L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9Ld8L_b-1','a')                                  
    txt.write(str(wgt_d9Ld8L[38]/number)+'\n')
    txt.close()     
    
    
    
    

    txt=open('./data/d8Ld9L','a')                                  
    txt.write(str(wgt_d8Ld9L[0]/number)+'\n')
    txt.close()         
    txt=open('./data/d8Ld9L_b1b1b1','a')                                  
    txt.write(str(wgt_d8Ld9L[1]/number)+'\n')
    txt.close()        
    txt=open('./data/d8Ld9L_a1b1b1','a')                                  
    txt.write(str(wgt_d8Ld9L[2]/number)+'\n')
    txt.close()     
    txt=open('./data/d8Ld9L_a1b1b1_0','a')                                  
    txt.write(str(wgt_d8Ld9L[3]/number)+'\n')
    txt.close()            
    txt=open('./data/d8Ld9L_a1b1b1_1','a')                                  
    txt.write(str(wgt_d8Ld9L[4]/number)+'\n')
    txt.close()           
    txt=open('./data/d8Ld9L_a1b1a1','a')                                  
    txt.write(str(wgt_d8Ld9L[5]/number)+'\n')
    txt.close()            
    txt=open('./data/d8Ld9L_b0','a')                                  
    txt.write(str(wgt_d8Ld9L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8Ld9L_b1','a')                                  
    txt.write(str(wgt_d8Ld9L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8Ld9L_b-1','a')                                  
    txt.write(str(wgt_d8Ld9L[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d9L2d8','a')                                  
    txt.write(str(wgt_d9L2d8[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9L2d8_b1b1b1','a')                                  
    txt.write(str(wgt_d9L2d8[1]/number)+'\n')
    txt.close()           
    txt=open('./data/d9L2d8_a1a1b1','a')                                  
    txt.write(str(wgt_d9L2d8[2]/number)+'\n')
    txt.close()    
    txt=open('./data/d9L2d8_b1a1b1','a')                                  
    txt.write(str(wgt_d9L2d8[3]/number)+'\n')
    txt.close()            
    txt=open('./data/d9L2d8+d8d9L2','a')                                  
    txt.write(str(wgt_d9L2d8[0]/number+wgt_d8d9L2[0]/number)+'\n')
    txt.close()      
    txt=open('./data/d9L2d8_b0','a')                                  
    txt.write(str(wgt_d9L2d8[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L2d8_b1','a')                                  
    txt.write(str(wgt_d9L2d8[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L2d8_b-1','a')                                  
    txt.write(str(wgt_d9L2d8[38]/number)+'\n')
    txt.close()     




    txt=open('./data/d8d9L2','a')                                  
    txt.write(str(wgt_d8d9L2[0]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d9L2_b1b1b1','a')                                  
    txt.write(str(wgt_d8d9L2[1]/number)+'\n')
    txt.close()           
    txt=open('./data/d8d9L2_a1b1a1','a')                                  
    txt.write(str(wgt_d8d9L2[2]/number)+'\n')
    txt.close()            
    txt=open('./data/d8d9L2_a1b1b1','a')                                  
    txt.write(str(wgt_d8d9L2[3]/number)+'\n')
    txt.close()             
    txt=open('./data/d8d9L2_b0','a')                                  
    txt.write(str(wgt_d8d9L2[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8d9L2_b1','a')                                  
    txt.write(str(wgt_d8d9L2[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8d9L2_b-1','a')                                  
    txt.write(str(wgt_d8d9L2[38]/number)+'\n')
    txt.close()     




    txt=open('./data/d9d8L2','a')                                  
    txt.write(str(wgt_d9d8L2[0]/number)+'\n')
    txt.close()            
    txt=open('./data/d9d8L2_b0','a')                                  
    txt.write(str(wgt_d9d8L2[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9d8L2_b1','a')                                  
    txt.write(str(wgt_d9d8L2[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9d8L2_b-1','a')                                  
    txt.write(str(wgt_d9d8L2[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d9d9L3','a')                                  
    txt.write(str(wgt_d9d9L3[0]/number)+'\n')
    txt.close()            
    txt=open('./data/d9d9L3_b0','a')                                  
    txt.write(str(wgt_d9d9L3[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9d9L3_b1','a')                                  
    txt.write(str(wgt_d9d9L3[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9d9L3_b-1','a')                                  
    txt.write(str(wgt_d9d9L3[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d9Ld9L2','a')                                  
    txt.write(str(wgt_d9Ld9L2[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9Ld9L2_b1b1','a')                                  
    txt.write(str(wgt_d9Ld9L2[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld9L2_a1b1','a')                                  
    txt.write(str(wgt_d9Ld9L2[2]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld9L2_b1a1','a')                                  
    txt.write(str(wgt_d9Ld9L2[3]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld9L2_a1a1','a')                                  
    txt.write(str(wgt_d9Ld9L2[4]/number)+'\n')
    txt.close()  
    txt=open('./data/d9Ld9L2+d9L2d9L','a')                                  
    txt.write(str(wgt_d9Ld9L2[0]/number+wgt_d9L2d9L[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9Ld9L2_b1b1_d9L2d9L','a')                                  
    txt.write(str(wgt_d9Ld9L2[1]*2/number)+'\n')
    txt.close()     
    txt=open('./data/d9Ld9L2_b1a1_d9L2d9L','a')                                  
    txt.write(str(wgt_d9Ld9L2[3]*2/number)+'\n')
    txt.close()         
    txt=open('./data/d9Ld9L2_b0','a')                                  
    txt.write(str(wgt_d9Ld9L2[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9Ld9L2_b1','a')                                  
    txt.write(str(wgt_d9Ld9L2[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9Ld9L2_b-1','a')                                  
    txt.write(str(wgt_d9Ld9L2[38]/number)+'\n')
    txt.close()     
    
    
    
    
    
    

    txt=open('./data/d9L2d9L','a')                                  
    txt.write(str(wgt_d9L2d9L[0]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L2d9L_b1b1','a')                                  
    txt.write(str(wgt_d9L2d9L[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9L_a1b1','a')                                  
    txt.write(str(wgt_d9L2d9L[2]/number)+'\n')
    txt.close()              
    txt=open('./data/d9L2d9L_b1a1','a')                                  
    txt.write(str(wgt_d9L2d9L[3]/number)+'\n')
    txt.close()              
    txt=open('./data/d9L2d9L_a1a1','a')                                  
    txt.write(str(wgt_d9L2d9L[4]/number)+'\n')
    txt.close()              
    txt=open('./data/d9L2d9L_b0','a')                                  
    txt.write(str(wgt_d9L2d9L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L2d9L_b1','a')                                  
    txt.write(str(wgt_d9L2d9L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L2d9L_b-1','a')                                  
    txt.write(str(wgt_d9L2d9L[38]/number)+'\n')
    txt.close()     




    txt=open('./data/d9L3d9','a')                                  
    txt.write(str(wgt_d9L3d9[0]/number)+'\n')
    txt.close()         
    txt=open('./data/d9L3d9_b0','a')                                  
    txt.write(str(wgt_d9L3d9[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L3d9_b1','a')                                  
    txt.write(str(wgt_d9L3d9[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L3d9_b-1','a')                                  
    txt.write(str(wgt_d9L3d9[38]/number)+'\n')
    txt.close()     
    
    
    
    
    

    txt=open('./data/d8d10L3','a')                                  
    txt.write(str(wgt_d8d10L3[0]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d10L3_b0','a')                                  
    txt.write(str(wgt_d8d10L3[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8d10L3_b1','a')                                  
    txt.write(str(wgt_d8d10L3[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8d10L3_b-1','a')                                  
    txt.write(str(wgt_d8d10L3[38]/number)+'\n')
    txt.close()     
    
    
    
    
    

    txt=open('./data/d8Ld10L2','a')                                  
    txt.write(str(wgt_d8Ld10L2[0]/number)+'\n')
    txt.close() 
    txt=open('./data/d8Ld10L2+d10L2d8L','a')                                  
    txt.write(str(wgt_d8Ld10L2[0]/number*2)+'\n')
    txt.close()     
    txt=open('./data/d8Ld10L2_b0','a')                                  
    txt.write(str(wgt_d8Ld10L2[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8Ld10L2_b1','a')                                  
    txt.write(str(wgt_d8Ld10L2[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8Ld10L2_b-1','a')                                  
    txt.write(str(wgt_d8Ld10L2[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d8L2d10L','a')                                  
    txt.write(str(wgt_d8L2d10L[0]/number)+'\n')
    txt.close()    
    txt=open('./data/d8L2d10L_b0','a')                                  
    txt.write(str(wgt_d8L2d10L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8L2d10L_b1','a')                                  
    txt.write(str(wgt_d8L2d10L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8L2d10L_b-1','a')                                  
    txt.write(str(wgt_d8L2d10L[38]/number)+'\n')
    txt.close()     
    
    
    
    

    txt=open('./data/d8L3d10','a')                                  
    txt.write(str(wgt_d8L3d10[0]/number)+'\n')
    txt.close()                 
    txt=open('./data/d8L3d10_b0','a')                                  
    txt.write(str(wgt_d8L3d10[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d8L3d10_b1','a')                                  
    txt.write(str(wgt_d8L3d10[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d8L3d10_b-1','a')                                  
    txt.write(str(wgt_d8L3d10[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d9d10L4','a')                                  
    txt.write(str(wgt_d9d10L4[0]/number)+'\n')
    txt.close()           
    txt=open('./data/d9d10L4_b0','a')                                  
    txt.write(str(wgt_d9d10L4[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9d10L4_b1','a')                                  
    txt.write(str(wgt_d9d10L4[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9d10L4_b-1','a')                                  
    txt.write(str(wgt_d9d10L4[38]/number)+'\n')
    txt.close()     

    
    
    
    
    
    txt=open('./data/d9Ld10L3','a')                                  
    txt.write(str(wgt_d9Ld10L3[0]/number)+'\n')
    txt.close()       
    txt=open('./data/d9Ld10L3_b0','a')                                  
    txt.write(str(wgt_d9Ld10L3[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9Ld10L3_b1','a')                                  
    txt.write(str(wgt_d9Ld10L3[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9Ld10L3_b-1','a')                                  
    txt.write(str(wgt_d9Ld10L3[38]/number)+'\n')
    txt.close()     

    
    
    
    txt=open('./data/d9L2d10L2','a')                                  
    txt.write(str(wgt_d9L2d10L2[0]/number)+'\n')
    txt.close()    
    txt=open('./data/d9L2d10L2+d10L2d9L2','a')                                  
    txt.write(str(wgt_d9L2d10L2[0]/number*2)+'\n')
    txt.close()        
    txt=open('./data/d9L2d10L2_b0','a')                                  
    txt.write(str(wgt_d9L2d10L2[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L2d10L2_b1','a')                                  
    txt.write(str(wgt_d9L2d10L2[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L2d10L2_b-1','a')                                  
    txt.write(str(wgt_d9L2d10L2[38]/number)+'\n')
    txt.close()     

    
    
    
    
    txt=open('./data/d9L3d10L','a')                                  
    txt.write(str(wgt_d9L3d10L[0]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L3d10L_b0','a')                                  
    txt.write(str(wgt_d9L3d10L[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L3d10L_b1','a')                                  
    txt.write(str(wgt_d9L3d10L[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L3d10L_b-1','a')                                  
    txt.write(str(wgt_d9L3d10L[38]/number)+'\n')
    txt.close()     
    
    
    
    

    txt=open('./data/d9L4d10','a')                                  
    txt.write(str(wgt_d9L4d10[0]/number)+'\n')
    txt.close()               
    txt=open('./data/d9L4d10_b0','a')                                  
    txt.write(str(wgt_d9L4d10[36]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L4d10_b1','a')                                  
    txt.write(str(wgt_d9L4d10[37]/number)+'\n')
    txt.close()       
    txt=open('./data/d9L4d10_b-1','a')                                  
    txt.write(str(wgt_d9L4d10[38]/number)+'\n')
    txt.close()     

    
    
    
    txt=open('./data/s','a')                                  
    txt.write(str(wgt_s[0]/number)+'\n')
    txt.close()           
    txt=open('./data/s_b0','a')                                  
    txt.write(str(wgt_s[36]/number)+'\n')
    txt.close()   
    txt=open('./data/s_b1','a')                                  
    txt.write(str(wgt_s[37]/number)+'\n')
    txt.close()       
    txt=open('./data/s_b-1','a')                                  
    txt.write(str(wgt_s[38]/number)+'\n')
    txt.close()     
    
    
    
    

    txt=open('./data/ds','a')                                  
    txt.write(str(wgt_ds[0]/number)+'\n')
    txt.close()     
    txt=open('./data/dssoo','a')                                  
    txt.write(str(wgt_ds[1]/number)+'\n')
    txt.close()         
    txt=open('./data/dsooo','a')                                  
    txt.write(str(wgt_ds[2]/number)+'\n')
    txt.close()    
    txt=open('./data/dssoo_2_22','a')                                  
    txt.write(str(wgt_ds[3]/number)+'\n')
    txt.close()     
    txt=open('./data/dssoo_2_20','a')                                  
    txt.write(str(wgt_ds[4]/number)+'\n')
    txt.close()      
    txt=open('./data/dssoo_2_00','a')                                  
    txt.write(str(wgt_ds[5]/number)+'\n')
    txt.close()      
    txt=open('./data/dsooo_2_220','a')                                  
    txt.write(str(wgt_ds[6]/number)+'\n')
    txt.close()     
    txt=open('./data/dsooo_2_200','a')                                  
    txt.write(str(wgt_ds[7]/number)+'\n')
    txt.close()      
    txt=open('./data/dsooo_2_000','a')                                  
    txt.write(str(wgt_ds[8]/number)+'\n')
    txt.close()      
    txt=open('./data/dssoo_2_22_b1','a')                                  
    txt.write(str(wgt_ds[9]/number)+'\n')
    txt.close()     
    txt=open('./data/dssoo_2_20_b1','a')                                  
    txt.write(str(wgt_ds[10]/number)+'\n')
    txt.close()      
    txt=open('./data/dssoo_2_00_b1','a')                                  
    txt.write(str(wgt_ds[11]/number)+'\n')
    txt.close()      
    txt=open('./data/dsooo_2_220_b1','a')                                  
    txt.write(str(wgt_ds[12]/number)+'\n')
    txt.close()     
    txt=open('./data/dsooo_2_200_b1','a')                                  
    txt.write(str(wgt_ds[13]/number)+'\n')
    txt.close()      
    txt=open('./data/dsooo_2_000_b1','a')                                  
    txt.write(str(wgt_ds[14]/number)+'\n')
    txt.close()         
    txt=open('./data/ds_b0','a')                                  
    txt.write(str(wgt_ds[36]/number)+'\n')
    txt.close()   
    txt=open('./data/ds_b1','a')                                  
    txt.write(str(wgt_ds[37]/number)+'\n')
    txt.close()       
    txt=open('./data/ds_b-1','a')                                  
    txt.write(str(wgt_ds[38]/number)+'\n')
    txt.close()     
    

    
    
    
    txt=open('./data/dds','a')                                  
    txt.write(str(wgt_dds[0]/number)+'\n')
    txt.close()  
    txt=open('./data/ddsso','a')                                  
    txt.write(str(wgt_dds[1]/number)+'\n')
    txt.close()          
    txt=open('./data/ddsoo','a')                                  
    txt.write(str(wgt_dds[2]/number)+'\n')
    txt.close() 
    txt=open('./data/ddsso_22','a')                                  
    txt.write(str(wgt_dds[3]/number)+'\n')
    txt.close()          
    txt=open('./data/ddsso_20','a')                                  
    txt.write(str(wgt_dds[4]/number)+'\n')
    txt.close()             
    txt=open('./data/ddsso_00','a')                                  
    txt.write(str(wgt_dds[5]/number)+'\n')
    txt.close()             
    txt=open('./data/ddsoo_22','a')                                  
    txt.write(str(wgt_dds[6]/number)+'\n')
    txt.close()          
    txt=open('./data/ddsoo_20','a')                                  
    txt.write(str(wgt_dds[7]/number)+'\n')
    txt.close()             
    txt=open('./data/ddsoo_00','a')                                  
    txt.write(str(wgt_dds[8]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_22_22_b1b1','a')                                  
    txt.write(str(wgt_dds[9]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_22_22_a1b1','a')                                  
    txt.write(str(wgt_dds[10]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_22_22_b1a1','a')                                  
    txt.write(str(wgt_dds[11]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_22_22_a1a1','a')                                  
    txt.write(str(wgt_dds[12]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_20_22_b1b1','a')                                  
    txt.write(str(wgt_dds[13]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_20_22_a1b1','a')                                  
    txt.write(str(wgt_dds[14]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_20_22_b1a1','a')                                  
    txt.write(str(wgt_dds[15]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_20_22_a1a1','a')                                  
    txt.write(str(wgt_dds[16]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_00_22_b1b1','a')                                  
    txt.write(str(wgt_dds[17]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_00_22_a1b1','a')                                  
    txt.write(str(wgt_dds[18]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_00_22_b1a1','a')                                  
    txt.write(str(wgt_dds[19]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_00_22_a1a1','a')                                  
    txt.write(str(wgt_dds[20]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_22_22_b1b1','a')                                  
    txt.write(str(wgt_dds[21]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_22_22_a1b1','a')                                  
    txt.write(str(wgt_dds[22]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_22_22_b1a1','a')                                  
    txt.write(str(wgt_dds[23]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_22_22_a1a1','a')                                  
    txt.write(str(wgt_dds[24]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_20_22_b1b1','a')                                  
    txt.write(str(wgt_dds[25]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_20_22_a1b1','a')                                  
    txt.write(str(wgt_dds[26]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_20_22_b1a1','a')                                  
    txt.write(str(wgt_dds[27]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_20_22_a1a1','a')                                  
    txt.write(str(wgt_dds[28]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_00_22_b1b1','a')                                  
    txt.write(str(wgt_dds[29]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_00_22_a1b1','a')                                  
    txt.write(str(wgt_dds[30]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_00_22_b1a1','a')                                  
    txt.write(str(wgt_dds[31]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_00_22_a1a1','a')                                  
    txt.write(str(wgt_dds[32]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsso_22_20_b1b1','a')                                  
    txt.write(str(wgt_dds[33]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_22_20_a1b1','a')                                  
    txt.write(str(wgt_dds[34]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_22_20_b1a1','a')                                  
    txt.write(str(wgt_dds[35]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_22_20_a1a1','a')                                  
    txt.write(str(wgt_dds[36]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_20_20_b1b1','a')                                  
    txt.write(str(wgt_dds[37]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_20_20_a1b1','a')                                  
    txt.write(str(wgt_dds[38]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_20_20_b1a1','a')                                  
    txt.write(str(wgt_dds[39]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_20_20_a1a1','a')                                  
    txt.write(str(wgt_dds[40]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_00_20_b1b1','a')                                  
    txt.write(str(wgt_dds[41]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsso_00_20_a1b1','a')                                  
    txt.write(str(wgt_dds[42]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsso_00_20_b1a1','a')                                  
    txt.write(str(wgt_dds[43]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_00_20_a1a1','a')                                  
    txt.write(str(wgt_dds[44]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_22_20_b1b1','a')                                  
    txt.write(str(wgt_dds[45]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_22_20_a1b1','a')                                  
    txt.write(str(wgt_dds[46]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_22_20_b1a1','a')                                  
    txt.write(str(wgt_dds[47]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_22_20_a1a1','a')                                  
    txt.write(str(wgt_dds[48]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_20_20_b1b1','a')                                  
    txt.write(str(wgt_dds[49]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_20_20_a1b1','a')                                  
    txt.write(str(wgt_dds[50]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_20_20_b1a1','a')                                  
    txt.write(str(wgt_dds[51]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_20_20_a1a1','a')                                  
    txt.write(str(wgt_dds[52]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_00_20_b1b1','a')                                  
    txt.write(str(wgt_dds[53]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_00_20_a1b1','a')                                  
    txt.write(str(wgt_dds[54]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsoo_00_20_b1a1','a')                                  
    txt.write(str(wgt_dds[55]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_00_20_a1a1','a')                                  
    txt.write(str(wgt_dds[56]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_22_22','a')                                  
    txt.write(str(wgt_dds[57]/number)+'\n')
    txt.close()          
    txt=open('./data/ddsso_22_20','a')                                  
    txt.write(str(wgt_dds[58]/number)+'\n')
    txt.close()          
    txt=open('./data/ddsso_20_22','a')                                  
    txt.write(str(wgt_dds[59]/number)+'\n')
    txt.close() 
    txt=open('./data/ddsso_20_20','a')                                  
    txt.write(str(wgt_dds[60]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsso_00_22','a')                                  
    txt.write(str(wgt_dds[61]/number)+'\n')
    txt.close() 
    txt=open('./data/ddsso_00_20','a')                                  
    txt.write(str(wgt_dds[62]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsoo_22_22','a')                                  
    txt.write(str(wgt_dds[63]/number)+'\n')
    txt.close() 
    txt=open('./data/ddsoo_22_20','a')                                  
    txt.write(str(wgt_dds[64]/number)+'\n')
    txt.close()      
    txt=open('./data/ddsoo_20_22','a')                                  
    txt.write(str(wgt_dds[65]/number)+'\n')
    txt.close()  
    txt=open('./data/ddsoo_20_20','a')                                  
    txt.write(str(wgt_dds[66]/number)+'\n')
    txt.close()      
    txt=open('./data/ddsoo_00_22','a')                                  
    txt.write(str(wgt_dds[67]/number)+'\n')
    txt.close()  
    txt=open('./data/ddsoo_00_20','a')                                  
    txt.write(str(wgt_dds[68]/number)+'\n')
    txt.close()  
    txt=open('./data/ddsso_b0','a')                                  
    txt.write(str(wgt_dds[73]/number)+'\n')
    txt.close()   
    txt=open('./data/ddsso_b1','a')                                  
    txt.write(str(wgt_dds[74]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsso_b-1','a')                                  
    txt.write(str(wgt_dds[75]/number)+'\n')
    txt.close()         
    txt=open('./data/ddsoo_b0','a')                                  
    txt.write(str(wgt_dds[76]/number)+'\n')
    txt.close()   
    txt=open('./data/ddsoo_b1','a')                                  
    txt.write(str(wgt_dds[77]/number)+'\n')
    txt.close()       
    txt=open('./data/ddsoo_b-1','a')                                  
    txt.write(str(wgt_dds[78]/number)+'\n')
    txt.close()     
    
    
    
    
    txt=open('./data/ddds','a')                                  
    txt.write(str(wgt_ddds[0]/number)+'\n')
    txt.close()           
    txt=open('./data/dddss','a')                                  
    txt.write(str(wgt_ddds[1]/number)+'\n')
    txt.close()         
    txt=open('./data/dddso','a')                                  
    txt.write(str(wgt_ddds[2]/number)+'\n')
    txt.close()               
    txt=open('./data/dddss_220','a')                                  
    txt.write(str(wgt_ddds[3]/number)+'\n')
    txt.close()   
    txt=open('./data/dddss_200','a')                                  
    txt.write(str(wgt_ddds[4]/number)+'\n')
    txt.close()           
    txt=open('./data/dddso_220','a')                                  
    txt.write(str(wgt_ddds[5]/number)+'\n')
    txt.close()   
    txt=open('./data/dddso_200','a')                                  
    txt.write(str(wgt_ddds[6]/number)+'\n')
    txt.close()             
    txt=open('./data/dddss_220_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[7]/number)+'\n')
    txt.close()        
    txt=open('./data/dddss_220_a1b1b1','a')                                  
    txt.write(str(wgt_ddds[8]/number)+'\n')
    txt.close()                
    txt=open('./data/dddss_220_a1b1a1','a')                                  
    txt.write(str(wgt_ddds[9]/number)+'\n')
    txt.close()       
    txt=open('./data/dddss_200_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[10]/number)+'\n')
    txt.close()        
    txt=open('./data/dddss_200_b1a1b1','a')                                  
    txt.write(str(wgt_ddds[11]/number)+'\n')
    txt.close()                
    txt=open('./data/dddss_200_a1a1b1','a')                                  
    txt.write(str(wgt_ddds[12]/number)+'\n')
    txt.close()              
    txt=open('./data/dddso_220_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[13]/number)+'\n')
    txt.close()        
    txt=open('./data/dddso_220_a1b1b1','a')                                  
    txt.write(str(wgt_ddds[14]/number)+'\n')
    txt.close()                
    txt=open('./data/dddso_220_a1b1a1','a')                                  
    txt.write(str(wgt_ddds[15]/number)+'\n')
    txt.close()       
    txt=open('./data/dddso_200_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[16]/number)+'\n')
    txt.close()        
    txt=open('./data/dddso_200_b1a1b1','a')                                  
    txt.write(str(wgt_ddds[17]/number)+'\n')
    txt.close()                
    txt=open('./data/dddso_200_a1a1b1','a')                                  
    txt.write(str(wgt_ddds[18]/number)+'\n')
    txt.close()          
    txt=open('./data/dddso_200_2','a')                                  
    txt.write(str(wgt_ddds[19]/number)+'\n')
    txt.close() 
    txt=open('./data/dddso_200_0','a')                                  
    txt.write(str(wgt_ddds[20]/number)+'\n')
    txt.close()     
    txt=open('./data/dddss_b0','a')                                  
    txt.write(str(wgt_ddds[73]/number)+'\n')
    txt.close()   
    txt=open('./data/dddss_b1','a')                                  
    txt.write(str(wgt_ddds[74]/number)+'\n')
    txt.close()       
    txt=open('./data/dddss_b-1','a')                                  
    txt.write(str(wgt_ddds[75]/number)+'\n')
    txt.close()         
    txt=open('./data/dddso_b0','a')                                  
    txt.write(str(wgt_ddds[76]/number)+'\n')
    txt.close()   
    txt=open('./data/dddso_b1','a')                                  
    txt.write(str(wgt_ddds[77]/number)+'\n')
    txt.close()       
    txt=open('./data/dddso_b-1','a')                                  
    txt.write(str(wgt_ddds[78]/number)+'\n')
    txt.close()      

    
    
    
    txt=open('./data/dddds','a')                                  
    txt.write(str(wgt_dddds[0]/number)+'\n')
    txt.close()   
    txt=open('./data/dddds_b1b1b1b1','a')                                  
    txt.write(str(wgt_dddds[1]/number)+'\n')
    txt.close()      
    txt=open('./data/dddds_a1b1a1b1','a')                                  
    txt.write(str(wgt_dddds[2]/number)+'\n')
    txt.close()        
    txt=open('./data/dddds_a1b1b1b1','a')                                  
    txt.write(str(wgt_dddds[3]/number)+'\n')
    txt.close()    
    txt=open('./data/dddds_b1b1a1b1','a')                                  
    txt.write(str(wgt_dddds[4]/number)+'\n')
    txt.close()        
    txt=open('./data/dddds_a1a1a1a1','a')                                  
    txt.write(str(wgt_dddds[5]/number)+'\n')
    txt.close()        
    txt=open('./data/dddds_b0','a')                                  
    txt.write(str(wgt_dddds[76]/number)+'\n')
    txt.close()   
    txt=open('./data/dddds_b1','a')                                  
    txt.write(str(wgt_dddds[77]/number)+'\n')
    txt.close()       
    txt=open('./data/dddds_b-1','a')                                  
    txt.write(str(wgt_dddds[78]/number)+'\n')
    txt.close()          
    
    

    txt=open('./data/number','a')                                  
    txt.write(str(number)+'\n')
    txt.close() 
        
        




    print("--- get_ground_state %s seconds ---" % (time.time() - t1))
                
    return vals, vecs 

#########################################################################
    # set up Lanczos solver
#     dim  = VS.dim
#     scratch = np.empty(dim, dtype = complex)
    
#     #`x0`: Starting vector. Use something randomly initialized
#     Phi0 = np.zeros(dim, dtype = complex)
#     Phi0[10] = 1.0
    
#     vecs = np.zeros(dim, dtype = complex)
#     solver = lanczos.LanczosSolver(maxiter = 200, 
#                                    precision = 1e-12, 
#                                    cond = 'UPTOMAX', 
#                                    eps = 1e-8)
#     vals = solver.lanczos(x0=Phi0, scratch=scratch, y=vecs, H=matrix)
#     print ('GS energy = ', vals)
    
#     # get state components in GS; note that indices is a tuple
#     indices = np.nonzero(abs(vecs)>0.01)
#     wgt_d8 = np.zeros(6)
#     wgt_d9L = np.zeros(4)
#     wgt_d10L2 = np.zeros(1)

#     print ("Compute the weights in GS (lowest Aw peak)")
#     #for i in indices[0]:
#     for i in range(0,len(vecs)):
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
 
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         orb1 = state['hole1_orb']
#         orb2 = state['hole2_orb']
#         orb3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']

#         #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
#         #    continue
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         o12 = sorted([orb1,orb2,orb3])
#         o12 = tuple(o12)

#         if i in indices[0]:
#             print (' state ', orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3 ,'S=',S12,'Sz=',Sz12,", weight = ", abs(vecs[i,k])**2)
#     return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
