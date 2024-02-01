'''
Useful subroutines 
'''
import subprocess
import os
import sys
import time
import shutil
import numpy
import scipy.sparse as sps
import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util

#####################################
def write_Aw(fname,Aw,w_vals):
    f = open('./data_Aw/'+fname,'w',1) 
    f.write('#omega\tspectral weight\n')
    for i in xrange(0,len(w_vals)):
        f.write('{:.6e}\t{:.6e}\n'.format(float(w_vals[i]),Aw[i]))
        
def write_GS(fname,A,ep,tpd,Egs):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_GS/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(A,ep,tpd,Egs))
    
def write_GS2(fname,A,ep,pds,pdp,Egs):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_GS/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(A,ep,pds,pdp,Egs))
    
def write_GS_components(fname,A,ep,tpd,wgt_d8, wgt_d9L, wgt_d10L2):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_GS/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'
            .format(A,ep,tpd, wgt_d8[0],wgt_d8[1],wgt_d8[2],wgt_d8[3],wgt_d8[4],wgt_d8[5],\
             wgt_d9L[0],wgt_d9L[1],wgt_d9L[2],wgt_d9L[3],wgt_d10L2[0]))

def write_GS_components2(fname,A,ep,pds,pdp,wgt_d8, wgt_d9L, wgt_d10L2):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_GS/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'
            .format(A,ep,pds,pdp, wgt_d8[0],wgt_d8[1],wgt_d8[2],wgt_d8[3],wgt_d8[4],wgt_d8[5],\
             wgt_d9L[0],wgt_d9L[1],wgt_d9L[2],wgt_d9L[3],wgt_d10L2[0]))
    
def write_lowpeak(fname,A,ep,tpd,w_peak,weight):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_lowpeak/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(A,ep,tpd,w_peak,weight))
    
def write_lowpeak2(fname,A,ep,pds,pdp,w_peak,weight):
    #"a" - Append - will append to the end of the file
    #"w" - Write - will overwrite any existing content
    f = open('./data_lowpeak/'+fname,'a',1) 
    f.write('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(A,ep,pds,pdp,w_peak,weight))
    
##################################################################
# def get_statistic_2orb(o1,o2):
#     '''
#     Get how many orbs are on Ni, O separately
#     and write info into dorbs and porbs
#     '''  
#     nNi_Cu = 0; nO = 0; dorbs=[]; porbs=[]
#     if o1 in pam.Ni_Cu_orbs:
#         nNi_Cu += 1; dorbs.append(o1)   
#     elif o1 in pam.O_orbs:
#         nO += 1; porbs.append(o1)
#     if o2 in pam.Ni_Cu_orbs:
#         nNi_Cu += 1; dorbs.append(o2)
#     elif o2 in pam.O_orbs:
#         nO += 1; porbs.append(o2)     

#     assert(nO ==len(porbs))
#     assert(nNi_Cu ==len(dorbs))
    
#     return nNi_Cu, nO, dorbs, porbs


def get_NiCu_layer_orbs(state):
    '''
    Get orbs in Ni and Cu layers separately
   
    '''  
    #state = VS.get_state(VS.lookup_tbl[i])
            
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    s3 = state['hole3_spin']
    s4 = state['hole4_spin']  
    s5 = state['hole5_spin'] 
    s6 = state['hole6_spin']     
    o1 = state['hole1_orb']
    o2 = state['hole2_orb']
    o3 = state['hole3_orb']
    o4 = state['hole4_orb']  
    o5 = state['hole5_orb']     
    o6 = state['hole6_orb']      
    x1, y1, z1 = state['hole1_coord']
    x2, y2, z2 = state['hole2_coord']
    x3, y3, z3 = state['hole3_coord']
    x4, y4, z4 = state['hole4_coord']
    x5, y5, z5 = state['hole5_coord']  
    x6, y6, z6 = state['hole6_coord']      

    ss = [s1,s2,s3,s4,s5,s6]
    os = [o1,o2,o3,o4,o5,o6]
    xs = [x1,x2,x3,x4,x5,x6]
    ys = [y1,y2,y3,y4,y5,y6]
    zs = [z1,z2,z3,z4,z5,z6]
    
    Ni_i = []

    
    Ni_layer = []; Cu_layer = [];Cu_i = [];apz_layer = [];apz_i = []
    for i in range(6):
        if zs[i]==2:
            Ni_layer.append(ss[i])
            Ni_layer.append(os[i])
            Ni_layer.append(xs[i])
            Ni_layer.append(ys[i])
            Ni_layer.append(zs[i])
            Ni_i.append(i)
            
        elif zs[i]==0:
            Cu_layer.append(ss[i])
            Cu_layer.append(os[i])
            Cu_layer.append(xs[i])
            Cu_layer.append(ys[i])
            Cu_layer.append(zs[i])
            Cu_i.append(i)
        elif zs[i]==1:
            apz_layer.append(ss[i])
            apz_layer.append(os[i])
            apz_layer.append(xs[i])
            apz_layer.append(ys[i])
            apz_layer.append(zs[i])
            apz_i.append(i)            
            
            

    #Ni_z,Cu_i represents the number on Ni,Cu
#     print(s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4)
#     print(Ni_layer,Cu_layer,Ni_i,Cu_i,len(Ni_layer)/5, len(Cu_layer)/5)
    
    return Ni_layer, len(Ni_layer)/5, Cu_layer, len(Cu_layer)/5,Ni_i,Cu_i,apz_layer, len(apz_layer)/5,apz_i  # /5 to print out real number of holes
        
def get_Number_NiCu(state):
    '''
    How many d orbitals are there on the Ni/Cu layer
   
    '''  
    #state = VS.get_state(VS.lookup_tbl[i])
            
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    s3 = state['hole3_spin']
    s4 = state['hole4_spin']  
    s5 = state['hole5_spin'] 
    s6 = state['hole6_spin']     
    o1 = state['hole1_orb']
    o2 = state['hole2_orb']
    o3 = state['hole3_orb']
    o4 = state['hole4_orb']  
    o5 = state['hole5_orb']     
    o6 = state['hole6_orb']      
    x1, y1, z1 = state['hole1_coord']
    x2, y2, z2 = state['hole2_coord']
    x3, y3, z3 = state['hole3_coord']
    x4, y4, z4 = state['hole4_coord']
    x5, y5, z5 = state['hole5_coord']  
    x6, y6, z6 = state['hole6_coord']      

    ss = [s1,s2,s3,s4,s5,s6]
    os = [o1,o2,o3,o4,o5,o6]
    xs = [x1,x2,x3,x4,x5,x6]
    ys = [y1,y2,y3,y4,y5,y6]
    zs = [z1,z2,z3,z4,z5,z6]
    
    Ni_i = []

    
    Cu_i = []

    
    for i in range(6):
        if zs[i]==2 and xs[i]==0 and ys[i]==0:
            Ni_i.append(i)
        elif zs[i]==0 and xs[i]==0 and ys[i]==0:
            Cu_i.append(i)
   

    #Ni_z,Cu_i represents the number on Ni,Cu
#     print(s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4)
#     print(Ni_layer,Cu_layer,Ni_i,Cu_i,len(Ni_layer)/5, len(Cu_layer)/5)
    
    return len(Ni_i),len(Cu_i)  # /5 to print out real number of holes             
    
# def get_statistic_orb(os):       
#     '''
#     get orb label, e.g. d9L or d9L2, for a given os = [01,02,...] (any length of os !!)
#     Assume that there is at least 1 hole, namely len(os)=1 or os is not empty
#     '''
#     Nos = len(os)
#     assert(Nos>0)
    
#     nNi_Cu = 0; nO = 0
#     dorbs=[]; porbs=[]
    
#     for i in range(Nos):
#         if os[i] in pam.Ni_Cu_orbs:
#             nNi_Cu += 1; dorbs.append(os[i])   
#         elif os[i] in pam.O_orbs:
#             nO += 1; porbs.append(os[i])
        
    
#     if nNi_Cu==1:
#         label = 'd9'
#     elif nNi_Cu==2:
#         label = 'd8'
        
#     if nO==1:
#         label = 'd9'
#     elif nO==2:
#         label = 'd10'
    
#     return nNi_Cu, nO, dorbs, porbs


def get_orb_edep(orb,z,epCu,epNi,epbilayer):
    '''
    resarch for orb's edep
    ''' 
    if orb in pam.Ni_Cu_orbs and z==2: 
        diag_el = pam.edNi[orb]
    elif orb in pam.Ni_Cu_orbs and z==0: 
        diag_el = pam.edCu[orb]  
    elif orb in pam.O_orbs and z==2: 
        diag_el = epNi
    elif orb in pam.O_orbs and z==0: 
        diag_el = epCu
    elif orb in pam.Obilayer_orbs and z==1: 
        diag_el = epbilayer        
    return diag_el

def get_double_append(i,n,s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4,s5,o5,x5,y5,z5,s6,o6,x6,y6,z6,\
                      d_list,p_list,idx,hole3456_part, double_part): 
    if o1 in pam.Ni_Cu_orbs and o2 in pam.Ni_Cu_orbs: #and not (o3 in pam.Ni_Cu_orbs and o4 in pam.Ni_Cu_orbs):
        d_list.append(i)
        idx.append(n); hole3456_part.append([s3, o3, x3, y3, z3,s4, o4, x4, y4, z4,s5, o5, x5, y5, z5,s6, o6, x6, y6, z6])
        double_part.append([s1,o1,x1,y1,z1,s2,o2,x2,y2,z2])
    elif o1 in pam.O_orbs and o2 in pam.O_orbs:
        p_list.append(i)

def lamlist(l1, l2, l3,l4,l5,l6):
    '''
    reduce the 'for' circulation
    '''     
    funs = []
    for i in l1: 
        for j in l2: 
            for k in l3:
                for h in l4:
                    for g in l5: 
                        for q in l6: 
                            x = lambda i=i, j=j, k=k, h=h, g=g, q=q: (i,j,k,h,g,q)
                            funs.append(x)

    return funs        
        
def lamlist1(l1, l2):
    '''
    reduce the 'for' circulation
    '''     
    funs = []
    for i in l1: 
        for j in l2: 
            x = lambda i=i, j=j: (i,j)
            funs.append(x)

    return funs         
# def get_d_double_3hole(VS, i):
#     '''
#     Determine which two holes are doubly occupancy for ith 3hole state
#     '''  
#     state = VS.get_state(VS.lookup_tbl[i])
#     s1 = state['hole1_spin']
#     s2 = state['hole2_spin']
#     s3 = state['hole3_spin']
#     o1 = state['hole1_orb']
#     o2 = state['hole2_orb']
#     o3 = state['hole3_orb']
#     x1, y1, z1 = state['hole1_coord']
#     x2, y2, z2 = state['hole2_coord']
#     x3, y3, z3 = state['hole3_coord']
    
#     # find out which two holes are on Ni/Cu
#     # idx is to label which hole is not on Ni/Cu
#     if (x1, y1, z1)==(x2, y2, z2):
#         Lspin=s3; Lorb=o3; Lpos=[x3, y3, z3]
#         idx=3
#         dpos = [x1, y1, z1]
#         o12 = sorted([o1,o2])
#         o12 = tuple(o12)
#     elif (x1, y1, z1)==(x3, y3, z3):
#         Lspin=s2; Lorb=o2; Lpos=[x2, y2, z2]
#         idx=2
#         s2 = s3
#         dpos = [x1, y1, z1]
#         o12 = sorted([o1,o3])
#         o12 = tuple(o12)
#     elif (x2, y2, z2)==(x3, y3, z3):
#         Lspin=s1; Lorb=o1; Lpos=[x1, y1, z1]
#         idx=1
#         s1 = s2
#         s2 = s3
#         dpos = [x2, y2, z2]
#         o12 = sorted([o2,o3])
#         o12 = tuple(o12)
            
#     return s1, s2, o12, dpos, idx, Lspin, Lorb, Lpos


def oppo_spin(s1):
    if s1=='up':
        so = 'dn'
    elif s1=='dn':
        so = 'up'
    
    return so

def check_dense_matrix_hermitian(matrix):
    '''
    Check if dense matrix is Hermitian. Returns True or False.
    '''
    dim = matrix.shape[0]
    out = True
    for row in range(0,dim):
        for col in range(0,dim):
            #if row==38 and col==85:
            #    print row, col, matrix[row,col], matrix[col,row]
            
            # sparse matrix has many zeros
            if abs(matrix[row,col])<1.e-10:
                continue
                
            if abs(matrix[row,col]-np.conjugate(matrix[col,row]))>1.e-10:
                print (row, col, matrix[row,col], matrix[col,row])
                out = False
                break
    return out

# def check_spin_group(row,col,data,VS):
#     '''
#     check if hoppings or interaction matrix occur within groups of (up,up), (dn,dn), and (up,dn) 
#     since (up,up) state cannot hop to a (up,dn) or (dn,dn) state
#     '''
#     out = True
#     dim = len(data)
#     assert(len(row)==len(col)==len(data))
    
#     for i in range(0,dim):
#         irow = row[i]
#         icol = col[i]
        
#         rstate = VS.get_state(VS.lookup_tbl[irow])
#         rs1 = rstate['hole1_spin']
#         rs2 = rstate['hole2_spin']
#         rs3 = rstate['hole3_spin']
#         cstate = VS.get_state(VS.lookup_tbl[icol])
#         cs1 = cstate['hole1_spin']
#         cs2 = cstate['hole2_spin']
#         cs3 = cstate['hole3_spin']
        
#         rs = sorted([rs1,rs2,rs3])
#         cs = sorted([cs1,cs2,cs3])
        
#         if rs!=cs:
#             ro1 = rstate['hole1_orb']
#             ro2 = rstate['hole2_orb']
#             ro3 = rstate['hole3_orb']
#             rx1, ry1 = rstate['hole1_coord']
#             rx2, ry2 = rstate['hole2_coord']
#             rx3, ry3 = rstate['hole3_coord']
            
#             co1 = cstate['hole1_orb']
#             co2 = cstate['hole2_orb']
#             co3 = cstate['hole2_orb']
#             cx1, cy1 = cstate['hole1_coord']
#             cx2, cy2 = cstate['hole2_coord']
#             cx3, cy3 = cstate['hole2_coord']
        
#             print ('Error:'+str(rs)+' hops to '+str(cs))
#             print ('Error occurs for state',irow,rs1,ro1,rx1,ry1,rs2,ro2,rx2,ry2,rs3,ro3,rx3,ry3,\
#                   'hops to state',icol,cs1,co1,cx1,cy1,cs2,co2,cx2,cy2,cs3,co3,cx3,cy3)
#             out = False
#             break
#     return out

def compare_matrices(m1,m2):
    '''
    Check if two matrices are the same. Returns True or False
    '''
    dim = m1.shape[0]
    if m2.shape[0] != dim:
        return False
    else:
        out = True
        for row in range(0,dim):
            for col in range(0,dim):
                if m1[row,col] != m2[row,col]:
                    out = False
                    break
        return out
    
def get_atomic_d8_energy(A,B,C):
    '''
    Atomic limite d8 energy
    '''
    E_1S = A+14*B+7*C
    E_1G = A+4*B+2*C
    E_1D = A-3*B+2*C
    E_3P = A+7*B
    E_3F = A-8*B
    print ("E_1S = ", E_1S)     
    print ("E_1G = ", E_1G)     
    print ("E_1D = ", E_1D) 
    print ("E_3P = ", E_3P)
    print ("E_3F = ", E_3F)
    
def plot_atomic_multiplet_peaks(data_for_maxval):
    maxval = max(data_for_maxval)
    yy = [0,maxval]
    xx = [pam.E_1S,pam.E_1S]
    plt.plot(xx, yy,'--k', linewidth=0.5)
    #text(pam.E_1S-0.2, 10.2, 'E_1S', fontsize=5)
    xx = [pam.E_1G,pam.E_1G]
    plt.plot(xx, yy,'--k', linewidth=0.5)
    #text(pam.E_1G-0.2, 10.5, 'E_1G', fontsize=5)
    xx = [pam.E_1D,pam.E_1D]
    plt.plot(xx, yy,'--k', linewidth=0.5)
    #text(pam.E_1D-0.2, 10.8, 'E_1D', fontsize=5)
    xx = [pam.E_3P,pam.E_3P]
    plt.plot(xx, yy,'--k', linewidth=0.5)
    #text(pam.E_3P-0.2, 11.1, 'E_3P', fontsize=5)
    xx = [pam.E_3F,pam.E_3F]
    plt.plot(xx, yy,'--k', linewidth=0.5)
    #text(pam.E_3F-0.2, 11.4, 'E_3F', fontsize=5)
    
def checkU_unitary(U,U_d):
    UdU = U_d.dot(U)
    sh = UdU.shape
    print (sh)
    bb = sps.identity(sh[0], format='coo')
    tmp = UdU-bb
    print ('U_d.dot(U)-I = ')
    for ii in range(0,sh[0]):
        for jj in range(0,sh[1]):
            if tmp[ii,jj]>1.e-10:
                print (tmp[ii,jj])
                
