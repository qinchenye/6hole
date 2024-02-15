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
    s6 = slabel[25]; orb6 = slabel[26]; x6 = slabel[27]; y6 = slabel[28]; z6 = slabel[29];     
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
                    
    tlabel = slabel3[20:25] + [s6,orb6,x6,y6,z6]
    tmp56 = reorder_z(tlabel)              
    if tmp56 == tlabel:
        slabel4 = slabel3 + [s6,orb6,x6,y6,z6]
    else:
        tlabel = slabel3[15:20] + [s6,orb6,x6,y6,z6] 
        tmp46 = reorder_z(tlabel)                  
        if tmp46 == tlabel:
            slabel4 = slabel3[0:20] + [s6,orb6,x6,y6,z6] + slabel3[20:25]
        else:
            tlabel = slabel3[10:15] + [s6,orb6,x6,y6,z6] 
            tmp36 = reorder_z(tlabel)                
            if tmp36 == tlabel:
                slabel4 = slabel3[0:15] + [s6,orb6,x6,y6,z6] + slabel3[15:25]   
            else:
                tlabel = slabel3[5:10] + [s6,orb6,x6,y6,z6] 
                tmp26 = reorder_z(tlabel)                        
                if tmp26 == tlabel:
                    slabel4 = slabel3[0:10] + [s6,orb6,x6,y6,z6] + slabel3[10:25]     
                else:
                    tlabel = slabel3[0:5] + [s6,orb6,x6,y6,z6] 
                    tmp16 = reorder_z(tlabel)                         
                    if tmp16 == tlabel:
                        slabel4 = slabel3[0:5] + [s6,orb6,x6,y6,z6] + slabel3[5:25]     
                    else: 
                        slabel4 = [s6,orb6,x6,y6,z6] + slabel3                    
                    
                    
                        
                
                
    return slabel4


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val):  
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
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
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
    
    
    
    
    
    
    if abs(vals[0]-vals[18])<10**(-5):
        number = 19
    elif abs(vals[0]-vals[17])<10**(-5):
        number = 18       
    elif abs(vals[0]-vals[16])<10**(-5):
        number = 17   
    elif abs(vals[0]-vals[15])<10**(-5):
        number = 16       
    elif abs(vals[0]-vals[14])<10**(-5):
        number = 15           
    elif abs(vals[0]-vals[13])<10**(-5):
        number = 14       
    elif abs(vals[0]-vals[12])<10**(-5):
        number = 13   
    elif abs(vals[0]-vals[11])<10**(-5):
        number = 12       
    elif abs(vals[0]-vals[10])<10**(-5):
        number = 11  
    elif abs(vals[0]-vals[9])<10**(-5):
        number = 10      
    elif abs(vals[0]-vals[8])<10**(-5):
        number = 9  
    elif abs(vals[0]-vals[7])<10**(-5):
        number = 8       
    elif abs(vals[0]-vals[6])<10**(-5):
        number = 7           
    elif abs(vals[0]-vals[5])<10**(-5):
        number = 6       
    elif abs(vals[0]-vals[4])<10**(-5):
        number = 5      
    elif abs(vals[0]-vals[3])<10**(-5):
        number = 4
    elif abs(vals[0]-vals[2])<10**(-5):
        number = 3        
    elif abs(vals[0]-vals[1])<10**(-5):
        number = 2
    else:
        number = 1
    print ('Degeneracy of ground state is ' ,number)      
    
#     wgt_LmLn = np.zeros(10)
#     wgt_d8d8L = np.zeros(20)
#     wgt_d8Ld8 = np.zeros(10)
#     wgt_d9Ld8L = np.zeros(20) 
#     wgt_d8Ld9L = np.zeros(20)         
#     wgt_d9L2d8= np.zeros(10)
#     wgt_d8d9L2= np.zeros(10)        
#     wgt_d9d8L2= np.zeros(10)
#     wgt_d9L3d9= np.zeros(10)   
#     wgt_d9L2d9L= np.zeros(10)  
#     wgt_d9Ld9L2= np.zeros(10)
#     wgt_d9d9L3= np.zeros(10)        
#     wgt_d8L3d10= np.zeros(10)   
#     wgt_d8L2d10L= np.zeros(10)  
#     wgt_d8Ld10L2= np.zeros(10)
#     wgt_d8d10L3= np.zeros(10) 
#     wgt_d9d10L4= np.zeros(10)
#     wgt_d9Ld10L3= np.zeros(10)        
#     wgt_d9L2d10L2= np.zeros(10)          
#     wgt_d9L3d10L= np.zeros(10)   
#     wgt_d9L4d10= np.zeros(10)  
    wgt_a1= np.zeros(10)   
    wgt_b1= np.zeros(10)       
    wgt_L= np.zeros(10) 
    wgt_O= np.zeros(10)      


    wgt_d8Ld8L= np.zeros(80) 
    wgt_d9L2d8L= np.zeros(80)     
    wgt_d9L2d9L2= np.zeros(80)     
    wgt_d9L2L3= np.zeros(80)     
    
#     wgt_s= np.zeros(10)         
#     wgt_ds= np.zeros(30)           
#     wgt_dds= np.zeros(80)    
#     wgt_ddds= np.zeros(80)           
#     wgt_dddds= np.zeros(10)         
    sumweight=0
    sumweight1=0
    synweight2=0
    
    
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
            s6 = state['hole6_spin']              
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb'] 
            orb5 = state['hole5_orb']  
            orb6 = state['hole6_orb']              
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']  
            x5, y5, z5 = state['hole5_coord']  
            x6, y6, z6 = state['hole6_coord']              

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Ni_12  = S_Ni_val[istate]
            Sz_Ni_12 = Sz_Ni_val[istate]
            S_Cu_12  = S_Cu_val[istate]
            Sz_Cu_12 = Sz_Cu_val[istate]
            
#             S_Niother_12  = S_other_Ni_val[i]
#             Sz_Niother_12 = Sz_other_Ni_val[i]
#             S_Cuother_12  = S_other_Cu_val[i]
#             Sz_Cuother_12 = Sz_other_Cu_val[i]

#             print ( i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,' ',orb5,s5,x5,y5,z5,\
#                '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
#                ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
#                ", weight = ", weight,'\n')                     

            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,s5,orb5,x5,y5,z5,s6,orb6,x6,y6,z6]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
            s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];     
            s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];                
            s6 = slabel[25]; orb6 = slabel[26]; x6 = slabel[27]; y6 = slabel[28]; z6 = slabel[29];       
    
      
    
    
            if weight >0.001:
                sumweight1=sumweight1+abs(vecs[istate,k])**2
                print (' state ', istate, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,' ',orb5,s5,x5,y5,z5,' ',orb6,s6,x6,y6,z6,\
                   '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                   ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
                   ", weight = ", weight,'\n')  
                
                
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
                
            if orb6=='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb6 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb6 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb6 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2                        
        
        
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and (orb6 in pam.O_orbs) and z1==z2==z5==2 and z3==z4==z6==0 : 
                wgt_d8Ld8L[0]+=abs(vecs[istate,k])**2        
                
                
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and (orb6 in pam.O_orbs) and z1==z4==z5==2 and z2==z3==z6==0 : 
                wgt_d9L2d8L[0]+=abs(vecs[istate,k])**2                        
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and (orb6 in pam.O_orbs) and z1==z2==z4==2 and z3==z5==z6==0 : 
                wgt_d9L2d8L[1]+=abs(vecs[istate,k])**2              
        

            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and (orb6 in pam.O_orbs) and z1==z3==z4==2 and z2==z5==z6==0 : 
                wgt_d9L2d9L2[0]+=abs(vecs[istate,k])**2
                
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and (orb6 in pam.O_orbs) and z1==z2==z3==2 and z4==z5==z6==0 : 
                wgt_d9L2L3[0]+=abs(vecs[istate,k])**2
                



#             if (orb1 in pam.O_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs)  and  (orb4 in pam.O_orbs) and  (orb5 in pam.O_orbs): 
#                 wgt_LmLn[0]+=abs(vecs[istate,k])**2 
  

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==z5==2 and z3==z4==0 : 
#                 wgt_d8Ld8[0]+=abs(vecs[istate,k])**2
#                 if orb1==orb2==orb3==orb4=='dx2y2':
#                     wgt_d8Ld8[1]+=abs(vecs[istate,k])**2                
#                 elif orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
#                     wgt_d8Ld8[2]+=abs(vecs[istate,k])**2  
#                 elif orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
#                     wgt_d8Ld8[3]+=abs(vecs[istate,k])**2                      
#                 elif orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
#                     wgt_d8Ld8[4]+=abs(vecs[istate,k])**2                        
                    

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
#                 wgt_d8d8L[0]+=abs(vecs[istate,k])**2             
#                 if orb1==orb2==orb3==orb4=='dx2y2':
#                     wgt_d8d8L[1]+=abs(vecs[istate,k])**2                
#                 elif orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
#                     wgt_d8d8L[2]+=abs(vecs[istate,k])**2                
#                 elif orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
#                     wgt_d8d8L[3]+=abs(vecs[istate,k])**2                      
#                 elif orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
#                     wgt_d8d8L[4]+=abs(vecs[istate,k])**2
                    
                    

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==2 and z2==z3==z5==0 : 
#                 wgt_d9Ld8L[0]+=abs(vecs[istate,k])**2
#                 if orb1==orb2==orb3=='dx2y2':
#                     wgt_d9Ld8L[1]+=abs(vecs[istate,k])**2
#                 elif orb1==orb3=='dx2y2' and orb2=='d3z2r2':
#                     wgt_d9Ld8L[2]+=abs(vecs[istate,k])**2                    
#                     if S_Cu_12 ==0 and S_Ni_12 ==0: 
#                         wgt_d9Ld8L[3]+=abs(vecs[istate,k])**2 
#                     elif S_Cu_12 ==1: 
#                         wgt_d9Ld8L[4]+=abs(vecs[istate,k])**2   
#                 elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
#                     wgt_d9Ld8L[5]+=abs(vecs[istate,k])**2                          
                
                
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z4==2 and z3==z5==0 : 
#                 wgt_d8Ld9L[0]+=abs(vecs[istate,k])**2 
#                 if orb1==orb2==orb3=='dx2y2':
#                     wgt_d8Ld9L[1]+=abs(vecs[istate,k])**2
#                 elif orb2==orb3=='dx2y2' and orb1=='d3z2r2':
#                     wgt_d8Ld9L[2]+=abs(vecs[istate,k])**2 
#                     if S_Ni_12 ==0 and S_Cu_12 ==0: 
#                         wgt_d8Ld9L[3]+=abs(vecs[istate,k])**2 
             
#                     elif S_Ni_12 ==1: 
#                         wgt_d8Ld9L[4]+=abs(vecs[istate,k])**2    
#                 elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
#                     wgt_d8Ld9L[5]+=abs(vecs[istate,k])**2  
                    
                    
                    
            
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==z5==2 and z2==z3==0 : 
#                 wgt_d9L2d8[0]+=abs(vecs[istate,k])**2 
#                 if orb1==orb2==orb3=='dx2y2':
#                     wgt_d9L2d8[1]+=abs(vecs[istate,k])**2
#                 elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
#                     wgt_d9L2d8[2]+=abs(vecs[istate,k])**2                  
#                 elif orb1==orb3=='dx2y2' and orb2=='d3z2r2':
#                     wgt_d9L2d8[3]+=abs(vecs[istate,k])**2                 
                
                
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
#                 wgt_d8d9L2[0]+=abs(vecs[istate,k])**2   
#                 if orb1==orb2==orb3=='dx2y2':
#                     wgt_d8d9L2[1]+=abs(vecs[istate,k])**2
#                 elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
#                     wgt_d8d9L2[2]+=abs(vecs[istate,k])**2                 
#                 elif orb2==orb3=='dx2y2' and orb1=='d3z2r2':
#                     wgt_d8d9L2[3]+=abs(vecs[istate,k])**2                 
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
#                 wgt_d9d8L2[0]+=abs(vecs[istate,k])**2                 

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
#                 wgt_d9d9L3[0]+=abs(vecs[istate,k])**2  
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==2 and z2==z4==z5==0 : 
#                 wgt_d9Ld9L2[0]+=abs(vecs[istate,k])**2  
#                 if orb1==orb2=='dx2y2':
#                     wgt_d9Ld9L2[1]+=abs(vecs[istate,k])**2
#                 elif orb1=='d3z2r2' and orb2=='dx2y2':
#                     wgt_d9Ld9L2[2]+=abs(vecs[istate,k])**2
#                 elif orb1=='dx2y2' and orb2=='d3z2r2':
#                     wgt_d9Ld9L2[3]+=abs(vecs[istate,k])**2                    
#                 elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                     wgt_d9Ld9L2[4]+=abs(vecs[istate,k])**2                    
                    
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==2 and z2==z5==0 : 
#                 wgt_d9L2d9L[0]+=abs(vecs[istate,k])**2    
#                 if orb1==orb2=='dx2y2':
#                     wgt_d9L2d9L[1]+=abs(vecs[istate,k])**2
#                 elif orb1=='d3z2r2' and orb2=='dx2y2':
#                     wgt_d9L2d9L[2]+=abs(vecs[istate,k])**2
#                 elif orb1=='dx2y2' and orb2=='d3z2r2':
#                     wgt_d9L2d9L[3]+=abs(vecs[istate,k])**2                    
#                 elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                     wgt_d9L2d9L[4]+=abs(vecs[istate,k])**2                    
                
                
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==z5==2 and z2==0 : 
#                 wgt_d9L3d9[0]+=abs(vecs[istate,k])**2                  

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
#                 wgt_d8d10L3[0]+=abs(vecs[istate,k])**2  
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==2 and z4==z5==0 : 
#                 wgt_d8Ld10L2[0]+=abs(vecs[istate,k])**2                  
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==2 and z5==0 : 
#                 wgt_d8L2d10L[0]+=abs(vecs[istate,k])**2                   
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==2 : 
#                 wgt_d8L3d10[0]+=abs(vecs[istate,k])**2                  
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==2 and z2==z3==z4==z5==0 : 
#                 wgt_d9d10L4[0]+=abs(vecs[istate,k])**2                     
                    
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==2 and z3==z4==z5==0 : 
#                 wgt_d9Ld10L3[0]+=abs(vecs[istate,k])**2                        
                    
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==2 and z4==z5==0 : 
#                 wgt_d9L2d10L2[0]+=abs(vecs[istate,k])**2                        
                    
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==2 and z5==0 : 
#                 wgt_d9L3d10L[0]+=abs(vecs[istate,k])**2    
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==2: 
#                 wgt_d9L4d10[0]+=abs(vecs[istate,k])**2                   
#             elif (orb1 in pam.Obilayer_orbs): 
#                 wgt_s[0]+=abs(vecs[istate,k])**2                      
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Obilayer_orbs): 
#                 wgt_ds[0]+=abs(vecs[istate,k])**2   
#                 if orb3 in pam.Obilayer_orbs:
#                     wgt_ds[1]+=abs(vecs[istate,k])**2 
#                     if z1==2 and z4==2 and z5==2:
#                         wgt_ds[3]+=abs(vecs[istate,k])**2*2 
#                         if orb1=='dx2y2':
#                             wgt_ds[9]+=abs(vecs[istate,k])**2*2  
#                     if z1==2 and z4==2 and z5==0:
#                         wgt_ds[4]+=abs(vecs[istate,k])**2*2 
#                         if orb1=='dx2y2':
#                             wgt_ds[10]+=abs(vecs[istate,k])**2 *2                         
#                     if z1==2 and z4==0 and z5==0:
#                         wgt_ds[5]+=abs(vecs[istate,k])**2*2                              
#                         if orb1=='dx2y2':
#                             wgt_ds[11]+=abs(vecs[istate,k])**2*2                          
                    
#                 elif orb3 in pam.O_orbs:
#                     wgt_ds[2]+=abs(vecs[istate,k])**2  
#                     if z1==2 and z3==2 and z4==2 and z5==0:
#                         wgt_ds[6]+=abs(vecs[istate,k])**2*2  
#                         if orb1=='dx2y2':
#                             wgt_ds[12]+=abs(vecs[istate,k])**2*2                               
#                     if z1==2 and z3==2 and z4==0 and z5==0:
#                         wgt_ds[7]+=abs(vecs[istate,k])**2*2   
#                         if orb1=='dx2y2':
#                             wgt_ds[13]+=abs(vecs[istate,k])**2*2                               
#                     if z1==2 and z3==0 and z4==0 and z5==0:
#                         wgt_ds[8]+=abs(vecs[istate,k])**2*2                         
#                         if orb1=='dx2y2':
#                             wgt_ds[14]+=abs(vecs[istate,k])**2 *2                          
                
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Obilayer_orbs): 
#                 wgt_dds[0]+=abs(vecs[istate,k])**2       
#                 if (orb4 in pam.Obilayer_orbs) and (orb5 in pam.O_orbs):
#                     wgt_dds[1]+=abs(vecs[istate,k])**2
#                     if z1==z2==2:
#                         wgt_dds[3]+=abs(vecs[istate,k])**2
#                         if z4==z5==2:
#                             wgt_dds[57]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[9]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[10]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[11]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[12]+=abs(vecs[istate,k])**2                              
#                         elif z4==2 and z5==0:
#                             wgt_dds[58]+=abs(vecs[istate,k])**2                           
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[33]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[34]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[35]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[36]+=abs(vecs[istate,k])**2       
                                
#                     if z1==2 and z2==0:
#                         wgt_dds[4]+=abs(vecs[istate,k])**2 
#                         if z4==z5==2:
#                             wgt_dds[59]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[13]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[14]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[15]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[16]+=abs(vecs[istate,k])**2                                
#                         elif z4==2 and z5==0:
#                             wgt_dds[60]+=abs(vecs[istate,k])**2                           
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[37]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[38]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[39]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[40]+=abs(vecs[istate,k])**2                          
#                     if z1==0 and z2==0:
#                         wgt_dds[5]+=abs(vecs[istate,k])**2   
#                         if z4==z5==2:
#                             wgt_dds[61]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[17]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[18]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[19]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[20]+=abs(vecs[istate,k])**2                                  
#                         elif z4==2 and z5==0:
#                             wgt_dds[62]+=abs(vecs[istate,k])**2                           
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[41]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[42]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[43]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[44]+=abs(vecs[istate,k])**2                          
                            
#                 if (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs):
#                     wgt_dds[2]+=abs(vecs[istate,k])**2        
#                     if z1==z2==2:
#                         wgt_dds[6]+=abs(vecs[istate,k])**2
#                         if z4==z5==2:
#                             wgt_dds[63]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[21]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[22]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[23]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[24]+=abs(vecs[istate,k])**2                              
#                         elif z4==2 and z5==0:
#                             wgt_dds[64]+=abs(vecs[istate,k])**2                            
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[45]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[46]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[47]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[48]+=abs(vecs[istate,k])**2  
#                     if z1==2 and z2==0:
#                         wgt_dds[7]+=abs(vecs[istate,k])**2 
#                         if z4==z5==2:
#                             wgt_dds[65]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[25]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[26]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[27]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[28]+=abs(vecs[istate,k])**2                                               
#                         elif z4==2 and z5==0:
#                             wgt_dds[66]+=abs(vecs[istate,k])**2                           
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[49]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[50]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[51]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[52]+=abs(vecs[istate,k])**2      
                                
#                     if z1==0 and z2==0:
#                         wgt_dds[8]+=abs(vecs[istate,k])**2  
#                         if z4==z5==2:
#                             wgt_dds[67]+=abs(vecs[istate,k])**2
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[29]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[30]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[31]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[32]+=abs(vecs[istate,k])**2                                 
#                         elif z4==2 and z5==0:
#                             wgt_dds[68]+=abs(vecs[istate,k])**2                           
#                             if orb1==orb2=='dx2y2':
#                                 wgt_dds[53]+=abs(vecs[istate,k])**2
#                             elif orb1=='d3z2r2' and orb2=='dx2y2':
#                                 wgt_dds[54]+=abs(vecs[istate,k])**2
#                             elif orb1=='dx2y2' and orb2=='d3z2r2':
#                                 wgt_dds[55]+=abs(vecs[istate,k])**2                    
#                             elif orb1=='d3z2r2' and orb2=='d3z2r2':
#                                 wgt_dds[56]+=abs(vecs[istate,k])**2                       
                    
                    
                    

#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Obilayer_orbs): 
#                 wgt_ddds[0]+=abs(vecs[istate,k])**2  
#                 if orb5 in pam.Obilayer_orbs:
#                     wgt_ddds[1]+=abs(vecs[istate,k])**2 
#                     if z1==z2==2 and z3==0:
#                         wgt_ddds[3]+=abs(vecs[istate,k])**2
#                         if orb1==orb2==orb3=='dx2y2':
#                             wgt_ddds[7]+=abs(vecs[istate,k])**2 
#                         elif orb1=='d3z2r2' and orb2==orb3=='dx2y2':
#                             wgt_ddds[8]+=abs(vecs[istate,k])**2                             
#                         elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
#                             wgt_ddds[9]+=abs(vecs[istate,k])**2  
                           
                            
                            
#                     if z1==2 and z2==z3==0:
#                         wgt_ddds[4]+=abs(vecs[istate,k])**2                                                  
#                         if orb1==orb2==orb3=='dx2y2':
#                             wgt_ddds[10]+=abs(vecs[istate,k])**2 
#                         elif orb2=='d3z2r2' and orb1==orb3=='dx2y2':
#                             wgt_ddds[11]+=abs(vecs[istate,k])**2                             
#                         elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
#                             wgt_ddds[12]+=abs(vecs[istate,k])**2 
                            
#                 if orb5 in pam.O_orbs:
#                     wgt_ddds[2]+=abs(vecs[istate,k])**2                 
#                     if z1==z2==2 and z3==0:
#                         wgt_ddds[5]+=abs(vecs[istate,k])**2  
#                         if orb1==orb2==orb3=='dx2y2':
#                             wgt_ddds[13]+=abs(vecs[istate,k])**2 
#                         elif orb1=='d3z2r2' and orb2==orb3=='dx2y2':
#                             wgt_ddds[14]+=abs(vecs[istate,k])**2                             
#                         elif orb1==orb3=='d3z2r2' and orb2=='dx2y2':
#                             wgt_ddds[15]+=abs(vecs[istate,k])**2  
#                         if z5==2:
#                             wgt_ddds[19]+=abs(vecs[istate,k])**2*2  
#                         if z5==0:
#                             wgt_ddds[20]+=abs(vecs[istate,k])**2*2                             
                        
#                     if z1==2 and z2==z3==0:
#                         wgt_ddds[6]+=abs(vecs[istate,k])**2                       
#                         if orb1==orb2==orb3=='dx2y2':
#                             wgt_ddds[16]+=abs(vecs[istate,k])**2 
#                         elif orb2=='d3z2r2' and orb1==orb3=='dx2y2':
#                             wgt_ddds[17]+=abs(vecs[istate,k])**2                             
#                         elif orb1==orb2=='d3z2r2' and orb3=='dx2y2':
#                             wgt_ddds[18]+=abs(vecs[istate,k])**2                 
                
#             elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.Obilayer_orbs): 
#                 wgt_dddds[0]+=abs(vecs[istate,k])**2                  
#                 if orb1==orb2==orb3==orb4=='dx2y2':
#                     wgt_dddds[1]+=abs(vecs[istate,k])**2  
#                 if orb1==orb3=='d3z2r2' and orb2==orb4=='dx2y2':
#                     wgt_dddds[2]+=abs(vecs[istate,k])**2                  
#                 if orb1=='d3z2r2' and orb2==orb3==orb4=='dx2y2':
#                     wgt_dddds[3]+=abs(vecs[istate,k])**2   
#                 if orb3=='d3z2r2' and orb1==orb2==orb4=='dx2y2':
#                     wgt_dddds[4]+=abs(vecs[istate,k])**2                     
#                 if orb1==orb2==orb3==orb4=='d3z2r2':
#                     wgt_dddds[5]+=abs(vecs[istate,k])**2                      

#             sumweight=sumweight+abs(vecs[istate,k])**2

#     print ('sumweight=',sumweight/number)
#     print ('LmLn=',wgt_LmLn[0]/number)
#     print ('d8Ld8=',wgt_d8Ld8[0]/number)
#     print ('d8d8L=',wgt_d8d8L[0]/number)       
#     print ('d9Ld8L=',wgt_d9Ld8L[0]/number) 
#     print ('d8Ld9L=',wgt_d8Ld9L[0]/number)         
#     print ('d9d8L2=',wgt_d9d8L2[0]/number)          
#     print ('d9L2d8=',wgt_d9L2d8[0]/number) 
#     print ('d8d9L2=',wgt_d8d9L2[0]/number)         
#     print ('d9L3d9=',wgt_d9L3d9[0]/number) 
#     print ('d9L2d9L=',wgt_d9L2d9L[0]/number)         
#     print ('d9Ld9L2=',wgt_d9Ld9L2[0]/number)         
#     print ('d9d9L3=',wgt_d9d9L3[0]/number)         
#     print ('d8L3d10=',wgt_d8L3d10[0]/number) 
#     print ('d8L2d10L=',wgt_d8L2d10L[0]/number)         
#     print ('d8Ld10L2=',wgt_d8Ld10L2[0]/number)         
#     print ('d8d10L3=',wgt_d8d10L3[0]/number)  
#     print ('d9d10L4=',wgt_d9d10L4[0]/number)          
#     print ('d9Ld10L3=',wgt_d9Ld10L3[0]/number)         
#     print ('d9L2d10L2=',wgt_d9L2d10L2[0]/number)         
#     print ('d9L3d10L=',wgt_d9L3d10L[0]/number) 
#     print ('d9L4d10=',wgt_d9L4d10[0]/number) 
#     print ('s=',wgt_s[0]/number)         
#     print ('ds=',wgt_ds[0]/number)             
#     print ('ddsso=',wgt_dds[1]/number)  
#     print ('ddsoo=',wgt_dds[2]/number)          
#     print ('dddss=',wgt_ddds[1]/number)  
#     print ('dddso=',wgt_ddds[2]/number)          
#     print ('dddds=',wgt_dddds[0]/number)          
    
    

    print ('d8Ld8L=',wgt_d8Ld8L[0]/number)   
    print ('d9L2d8L=',wgt_d9L2d8L[0]/number) 
    print ('d8Ld9L2=',wgt_d9L2d8L[1]/number) 
    print ('d9L2d9L2=',wgt_d9L2d9L2[0]/number)     
    print ('d9L2L3=',wgt_d9L2L3[0]/number)     

#     sumweight2 = wgt_LmLn[0]+wgt_d8Ld8[0]+wgt_d8d8L[0]+wgt_d9Ld8L[0]+wgt_d8Ld9L[0]+wgt_d9d8L2[0]+wgt_d9L2d8[0]+wgt_d8d9L2[0]\
#              +wgt_d9L3d9[0]+wgt_d9L2d9L[0]+wgt_d9Ld9L2[0]+wgt_d9d9L3[0]+wgt_d8L3d10[0]+wgt_d8L2d10L[0]\
#              +wgt_d8Ld10L2[0]+wgt_d8d10L3[0]+wgt_d9d10L4[0]+wgt_d9Ld10L3[0]+wgt_d9L2d10L2[0]+wgt_d9L3d10L[0]+wgt_d9L4d10[0]
#     print ('sumweight2=',sumweight2/number)
#         print ('s11=',s11)        
#         print ('s10=',s10)       
#         print ('s01=',s01)  
#         print ('s00=',s00)          







    path = './data'		# create file

    if os.path.isdir(path) == False:
        os.mkdir(path) 
        
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
    
    txt=open('./data/d8Ld8L','a')                                  
    txt.write(str(wgt_d8Ld8L[0]/number)+'\n')
    txt.close()      
    
    txt=open('./data/d9L2d8L','a')                                  
    txt.write(str(wgt_d9L2d8L[0]/number)+'\n')
    txt.close()          
    
    txt=open('./data/d9L2d9L2','a')                                  
    txt.write(str(wgt_d9L2d9L2[0]/number)+'\n')
    txt.close()          
    
    txt=open('./data/d9L2L3','a')                                  
    txt.write(str(wgt_d9L2L3[0]/number)+'\n')
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
