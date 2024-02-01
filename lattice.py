'''
Cu site (only one) and O square lattice surrounding it. 
Keep using CuO2 type lattice but now there is only (0,0) Cu-site
'''
import parameters as pam

# below used for get_uid and get_state in VS
if pam.Norb==4:
    orb_int = {'dx2y2': 0,\
               'px':    1,\
               'py':    2,\
               'apz':    3} 
    int_orb = {0: 'dx2y2',\
               1: 'px',\
               2: 'py',\
               3: 'apz'}
elif pam.Norb==5:
    orb_int = {'d3z2r2': 0,\
               'dx2y2': 1,\
               'px':    2,\
               'py':    3,\
               'apz':    4} 
    int_orb = {0: 'd3z2r2',\
               1: 'dx2y2',\
               2: 'px',\
               3: 'py',\
               4: 'apz'}   
    
    
elif pam.Norb==8:
    orb_int = {'d3z2r2': 0,\
               'dx2y2':  1,\
               'dxy':    2,\
               'dxz':    3,\
               'dyz':    4,\
               'px':     5,\
               'py':    6,\
               'apz':    7} 
    int_orb = {0: 'd3z2r2',\
               1: 'dx2y2',\
               2: 'dxy',\
               3: 'dxz',\
               4: 'dyz',\
               5: 'px',\
               6: 'py',\
               7: 'apz'} 

    
# apz means apical oxygen pz locating above Cu atom:

spin_int = {'up': 1,\
            'dn': 0}
int_spin = {1: 'up',\
            0: 'dn'} 

def get_unit_cell_rep(x,y,z):
    '''
    Given a vector (x,y) return the correpsonding orbital.

    Parameters
    -----------
    x,y: (integer) x and y component of vector pointing to a lattice site.
    
    Returns
    -------
    orbital: One of the following strings 'dx2y2', 
            'Ox1', 'Ox2', 'Oy1', 'Oy2', 'NotOnSublattice'
    '''
    # Note that x, y, z can be negative
    if x==0 and y==0 and z==0:
        return pam.Ni_Cu_orbs
    elif x==0 and y==0 and z==2:                                           #z=1 is Ni,z=0 is Cu
        return pam.Ni_Cu_orbs
    elif x==0 and y==0 and z==1:                                           #z=1 is Ni,z=0 is Cu
        return pam.Obilayer_orbs   
    elif abs(x) % 2 == 1 and abs(y) % 2 == 0 and (z==0 or z==2):
        return pam.O1_orbs
    elif abs(x) % 2 == 0 and abs(y) % 2 == 1 and (z==0 or z==2):
        return pam.O2_orbs
    else:
        return ['NotOnSublattice']
