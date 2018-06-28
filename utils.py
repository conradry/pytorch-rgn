import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import bcolz
from shutil import copyfile
from PIL import Image
import Bio.PDB as bio
import scipy
import torch
#import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import Dataset
from keras.utils.np_utils import to_categorical

residue_letter_codes = {'GLY': 'G','PRO': 'P','ALA': 'A','VAL': 'V','LEU': 'L',
                        'ILE': 'I','MET': 'M','CYS': 'C','PHE': 'F','TYR': 'Y',
                        'TRP': 'W','HIS': 'H','LYS': 'K','ARG': 'R','GLN': 'Q',
                        'ASN': 'N','GLU': 'E','ASP': 'D','SER': 'S','THR': 'T'}

aa2ix= {'G': 0,'P': 1,'A': 2,'V': 3,'L': 4,
          'I': 5,'M': 6,'C': 7,'F': 8,'Y': 9,
          'W': 10,'H': 11,'K': 12,'R': 13,'Q': 14,
          'N': 15,'E': 16,'D': 17,'S': 18,'T': 19}

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]
    
#get sequences and corresponding chain ids
def gather_sequence(pdb_id, seq_file):
    seqs=[]
    #chains=[]
    #get line numbers of pdb_id
    for ix, line in enumerate(seq_file):
        pos = np.core.defchararray.find(line, pdb_id)
        if pos > 0:
            seqs.append(seq_file[ix+1][:-1]) #cut off newline character
            #chains.append(line[pos+5]) #gives the chain letter from the line        
        
    return seqs

def create_targets(pdb_file_path):
    p = bio.PDBParser()
    s = p.get_structure('X', pdb_file_path)
    
    #first = []
    #last = []
    #coords = []
    #chains=[]
    chain_coords=[]
    
    #randomly select one model from the pdb file
    gen = s.get_models()
    l = list(gen)
    mod = l[np.random.randint(0, len(l))]
    
    #for model in s:
    for chain in mod:
        for ix,residue in enumerate(chain):
            coords = []
            #if ix == 0:
            #    first.append(residue.get_id()[1])
            if residue.get_id()[0] == ' ':
                #l = residue.get_id()[1]
                for atom in residue:
                    if atom.get_name() == "N":
                        n = atom.get_coord()
                    elif atom.get_name() == "CA":
                        ca = atom.get_coord()
                    elif atom.get_name() == "C":
                        cp = atom.get_coord()
                try: #in some cases N is missing on the first residue, so we append zeros instead
                    coords.append(np.stack([n,ca,cp], axis=0))
                    del n, ca, cp
                except:
                    #first[-1] += 1 move past the first residue and ignore it
                    pass
                    #coords.append(np.zeros(3,3))
        chain_coords.append(coords)
        #coords = []
        #chains.append(chain.get_id())
        #last.append(l)
    #final array is size 5x(no of residues)*3
    return chain_coords #, chains, first, last

def encode_sequence(sequence, onehot=True):
    vec=[]
    for chain in sequence:
        for c in chain:
            for aa, val in aa2ix.iteritems():
                if c == aa:
                    vec.append(val)
    if onehot:
        encoding = to_categorical(vec, 20)
        return np.uint8(encoding)
    
    return np.uint8(vec)

def parse_pdb(pdb_file, residue_letter_codes):
    #pdb_file = 'pdb5l6t.ent' #np.random.choice(pdb_list)
    p = bio.PDBParser()
    s = p.get_structure('X', pdb_path+pdb_file)
    
    gen = s.get_models()
    l = list(gen)
    mod = l[np.random.randint(0, len(l))] #choose random model when more than 1 exists
    
    seq_strs = []
    seq_locs = []
    for chain in mod:
        seq_str = ''
        seq_loc = []
        for residue in chain:
            if residue.get_id()[0] == ' ':
                letter_code = residue_letter_codes[residue.get_resname()]
                seq_str += letter_code
                for atom in residue:
                    seq_loc.append(atom.get_full_id()[3][1])
        seq_strs.append(seq_str)
        seq_locs.append(np.unique(seq_loc))
        
    return seq_strs, seq_locs

def align_indices(seq_strs, seq_locs, gt_seq, start_match_length=5):
    fill_indices = []
    for ix, pdb_seq in enumerate(seq_strs):
        search_seq = gt_seq[ix]
        pos = np.core.defchararray.find(search_seq, pdb_seq[:start_match_length])
        if pos < 0:
            raise ValueError('First 5 residues in pdb file have no match!')
        locs = seq_locs[ix] + (pos - seq_locs[ix][0])
        fill_indices.append(np.intersect1d(range(len(search_seq)), locs))
    
    return fill_indices

def calc_dist(atom1_coord, atom2_coord):
    return scipy.spatial.distance.euclidean(atom1_coord, atom2_coord)

def gt_dihedral_angles(pdb_file_path):
    p = bio.PDBParser()
    s = p.get_structure('X', pdb_file_path)
    calc_di = bio.vectors.calc_dihedral
    calc_ba = bio.vectors.calc_angle
    
    #torsional angles
    phi = []
    psi = []
    omega = []
    #bond angles
    ca_c_n = []
    c_n_ca = []
    n_ca_c = []
    #bond_lengths
    c_n = []
    n_ca = []
    ca_c = []
    
    for model in s:
        for chain in model:
            for ix, residue in enumerate(chain):
                for atom in residue:
                    if atom.get_name() == "N":
                        n = atom.get_vector()
                        n_coord = atom.get_coord()
                        if ix != 0:
                            psi.append(calc_di(np, cap, cp, n))
                            ca_c_n.append(calc_ba(cap, cp, n))
                            c_n.append(calc_dist(cp_coord, n_coord))
                    if atom.get_name() == "CA":
                        ca = atom.get_vector()
                        ca_coord = atom.get_coord()
                        if ix != 0:
                            omega.append(calc_di(cap, cp, n, ca))
                            c_n_ca.append(calc_ba(cp, n, ca))
                            n_ca.append(calc_dist(n_coord, ca_coord))
                    if atom.get_name() == "C":
                        c = atom.get_vector()
                        c_coord = atom.get_coord()
                        if ix != 0:
                            phi.append(calc_di(cp, n, ca, c))
                            n_ca_c.append(calc_ba(n, ca, c))
                            ca_c.append(calc_dist(ca_coord, c_coord))
                #store previous vectors
                np, cap, cp = n, ca, c
                cp_coord = c_coord

    torsional_angles = torch.stack([torch.tensor(psi), 
                                    torch.tensor(omega), 
                                    torch.tensor(phi)], dim=1)
    
    bond_angles = torch.stack([torch.tensor(ca_c_n), 
                               torch.tensor(c_n_ca), 
                               torch.tensor(n_ca_c)], dim=1)
    
    bond_lengths = torch.stack([torch.tensor(c_n), 
                                torch.tensor(n_ca), 
                                torch.tensor(ca_c)], dim=1)
        
    return torsional_angles, bond_angles, bond_lengths
