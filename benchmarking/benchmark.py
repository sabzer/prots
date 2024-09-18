###IMPORTS####


import copy
import sys
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
import time
import cProfile
import json
import pickle
from Bio.PDB import *
import os
from collections import defaultdict
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq3
from Bio.PDB.Polypeptide import *


import sw_functions as sw
import numpy as np
import jax #https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
import jax.numpy as jnp
import os
import json
import csv



###LDDT FUNCTIONS###

#for a list x, returns mtx st mtx[i,j] is pairwise distance of x[i] and x[j]
def pw(x):
    '''compute pairwise distance'''
    x_norm = np.square(x).sum(-1)
    xx = np.einsum("...ia,...ja->...ij",x,x)
    sq_dist = x_norm[...,:,None] + x_norm[...,None,:] - 2 * xx
    np.fill_diagonal(sq_dist,0)
    return np.sqrt(sq_dist+ 1e-10)


def distMtx(allCoordCA,distThreshold):
    dists=pw(allCoordCA)
    return dists

def reducedDistMtx(alnMtx,distMtx1,distMtx2):
    alnMtx=np.where(alnMtx>0.95,1,0)
    dim=np.sum(alnMtx)
    dim=int(dim)
    redDistMtx1=np.zeros((dim,dim))
    redDistMtx2=np.zeros((dim,dim))
    list1=[]
    list2=[]
    for i in range(np.shape(alnMtx)[0]):
        for j in range(np.shape(alnMtx)[1]):
            if alnMtx[i,j]==1:
                list1.append(i)
                list2.append(j)

    for l in range(len(list1)):
        for k in range(len(list1)):
            redDistMtx1[l,k]=distMtx1[list1[l],list1[k]]
            redDistMtx2[l,k]=distMtx2[list2[l],list2[k]]
    return redDistMtx1,redDistMtx2



#does prot1 as ref
def detLDDT(prot1,prot2,aln):
    
    allCoordCA1=allProtCACoord[prot1]
    allCoordCA2=allProtCACoord[prot2]
    
    distMtx1=distMtx(allCoordCA1,15)
    distMtx2=distMtx(allCoordCA2,15)
    
    reducedDistMtx1,reducedDistMtx2=reducedDistMtx(aln,distMtx1,distMtx2)
    
    thresh=[.5,1,2,4]
    threshV=[0,0,0,0]
    
    mask=np.where(reducedDistMtx1<15,1,0)-np.eye(np.shape(reducedDistMtx1)[0])
    
    relevantDist=(abs(reducedDistMtx1-reducedDistMtx2))
    for i in range(len(thresh)):
        threshV[i]=np.sum(np.where(relevantDist<thresh[i],1,0)*mask)

            
    denom=4*np.sum(np.where(distMtx1<15,1,0)-np.eye(np.shape(distMtx1)[0]))
    num=np.sum(np.array(threshV))
    
    if denom==0:
        return 0
    return num/denom
                


### FILE PARSING ###


def simMtxBlosum(pair):
    mtx1=allProtsClusteredDihedrals_1Hot[pair[0]]
    mtx2=allProtsClusteredDihedrals_1Hot[pair[1]]   
    
    p1=np.einsum('ik,kj->ij', mtx1, blosum) 
    simMtx=np.einsum('ik,kj->ij', p1, np.transpose(mtx2))

    simMtx=jnp.array(simMtx)
    
    len1=np.shape(mtx1)[0]
    len2=np.shape(mtx2)[0]
    lens=np.array([[len1,len2]])
    lens=jnp.array(lens)
    
    return (simMtx,lens)


def getLDDT(simMtx,lens,o=10):
    aln = my_affine_sw_func(simMtx, (lens[0][0],lens[0][1]), gap=-1,open=-1*o,temp=.001)
    lddt=detLDDT(pair[0],pair[1],aln)
    return lddt


### ACTUAL CODE###
if __name__ == "__main__":


    if len(sys.argv) <2:
        print("Usage: python benchmarking.py <number>")
        sys.exit(1)  # Exit the script indicating error
    try:
        protInd = int(sys.argv[1])

    except ValueError:
        print("Please provide a valid integer.")
        sys.exit(1)  # Exit the script indicating error

    print(f"The provided number is {protInd}")

    
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    my_sw_func = jax.jit(sw.sw(batch=False))
    my_affine_sw_func = jax.jit(sw.sw_affine(batch=False))
    my_nw_func = jax.jit(sw.nw(batch=False))


    num=protInd

    queryProts='queryProts.txt'
    refProts='referenceProts.txt'


    queryList=[]
    with open(queryProts, 'r') as file:
        for line in file:
            queryList.append(line.strip())

    refList=[]
    with open(refProts, 'r') as g:
        for line in g:
            refList.append(line.strip())

    relevantProt=queryList[num]


    allPairs=[]
    for prot in refList:
        allPairs.append((relevantProt,prot))


    with open("../variables/infoDictionaries/allProtCACoord.pkl","rb") as ff:
        allProtCACoord=pickle.load(ff)

    with open("../variables/infoDictionaries/allProtsClusteredDihedrals_1Hot.pkl", 'rb') as file:
        allProtsClusteredDihedrals_1Hot = pickle.load(file)  

    blosum=np.load('dihedralBlosum.npy')


    lddtDict={}
    print(num)
    print(relevantProt)
    for prot in refList:
        if prot!=relevantProt:
            pair=(relevantProt,prot)
            simMtx,lens=simMtxBlosum(pair)
            lddtDict[pair]=getLDDT(simMtx,lens)

    sorted_lddt_scores = sorted(lddtDict.items(), key=lambda x: x[1], reverse=True)

    filename = f"{relevantProt}.txt"
    with open(f"benchDihedral/{filename}", "w") as file:
        for pair, score in sorted_lddt_scores:
            file.write(f"{pair[0]} {pair[1]} {score}\n")