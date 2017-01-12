
# coding: utf-8

# # Tensor Factorization for Recommender Systems
# ### - CP factorization on tensor (user, music, context)
# ### - Written by ByungSoo Jeon, NAVER LABS

# In[59]:

# Add system path to use scikit-tensor Library
import sys
import sktensor as skt
import numpy as np
import math

# Set logging to DEBUG to see CP-ALS information
import logging
logging.basicConfig(level=logging.DEBUG)

# Use sparse matrix format
from scipy import sparse


# ### - Regularized matrix factorization

# In[60]:

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    e = 0
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
#                     for k in range(K):
#                         e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T, e


# ### - Regularized CP tensor factorization (Multiverse Recommendation, RecSys 10)

# In[61]:

def Regularized_CP_TF(X, A, B, C, R, steps=1000, alpha=0.0002, beta=0.02):
    e = 0
    for step in range(steps):
        nNnz = 0
        # Stochastic Gradient Descent(SGD) part
        for i in range(len(X)):
            for j in range(len(X[i])):
                for k in range(len(X[i][j])):
                    if X[i][j][k] > 0:
                        eijk = X[i][j][k] - np.dot(A[i,:],np.multiply(B[j,:],C[k,:]))
                        nNnz+=1
                        for r in range(R):
                            A[i][r] = A[i][r] + alpha * (2 * eijk * B[j][r] * C[k][r] - beta * A[i][r])
                            B[j][r] = B[j][r] + alpha * (2 * eijk * A[i][r] * C[k][r] - beta * B[j][r])
                            C[k][r] = C[k][r] + alpha * (2 * eijk * B[j][r] * A[i][r] - beta * C[k][r])                            
        
        # Stop condition for SGD
        e = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                for k in range(len(X[i][j])):
                    if X[i][j][k] > 0:
                        e = e + pow(X[i][j][k] - np.dot(A[i,:],np.multiply(B[j,:],C[k,:])), 2)
#                         for r in range(R):
#                             e = e + (beta/2) * (pow(A[i][r],2) + pow(B[j][r],2) + pow(C[k][r],2))
        # Convert error to RMSE
        e = math.sqrt(e/nNnz)
        if e < 0.3:
            print ("[TF for RS] Target RMSE(0.3) is reached.")
            break
        if (step%100==0) :
            print ("[TF for RS] "+str(step)+"/"+str(steps)+" steps done.")
    return A,B,C,e


# ### - Input utility matrix

# In[68]:

def Read_Utility_Matrix():
    # Read meta data to build dictionary in order to change matrix to tensor.
    metadata_file = open("full_meta_data.txt","r")
    lines = metadata_file.readlines()
    item_context_dic = {}
    nContext = 0

    # For now, context is Artist ID
    for line in lines:
        try:            
            item = int(line.split("\t")[0])
            context = int(line.split("\t")[2])
        except:
            pass
        else:
            item_context_dic[item] = context
            if (nContext < context+1) :
                nContext = context+1
            
    metadata_file.close()
    print ("[TF for RS] Reading Metadata done.")
    
    # Read matrix data
    rating_file = open("ratings.txt","r")
    lines = rating_file.readlines()
    users, items, ratings = [], [], []
    nUser, nItem = 0, 0
    
    for line in lines:
        user = int(line.split("\t")[0])
        item = int(line.split("\t")[1])
        rating = float(line.split("\t")[2])
        users.append(user)
        items.append(item)
        ratings.append(rating)
        if (nUser < user+1) :
            nUser = user+1
        if (nItem < item+1) :
            nItem = item+1
            
    rating_file.close()
#     utility_matrix = sparse.coo_matrix((ratings, (users, items)), shape=(nUser, nItem))
    X = np.zeros((nUser,nItem,nContext))
    for i in range(len(users)):
        if items[i] in item_context_dic:
            X[users[i]][items[i]][item_context_dic[items[i]]] = ratings[i]
    return X


# ### - Main function for CP

# In[69]:

if __name__ == "__main__":
#     X = [1,2,3,4,5,6,7,8,9,10,11,12]
#     X = np.array(X)
#     X = X.reshape(2,2,3)
    X = Read_Utility_Matrix()
    print ("[TF for RS] Reading ratings done.")
    R = 2

    A = np.random.rand(len(X),R)
    B = np.random.rand(len(X[0]),R)
    C = np.random.rand(len(X[0][0]),R)
    
    A, B, C, e = Regularized_CP_TF(X, A, B, C, R)
    print ("[TF for RS] RMSE : "+str(e))

#     print (A)
#     print (B)
#     print (C)
#     D = np.transpose(skt.khatrirao((C,B)))
#     print(np.matmul(A,D))
#     print(X)


# ### - Naive CP tensor factorization 

# In[53]:

# if __name__ == "__main__":
#     # Load Matlab data
#     X = Read_Utility_Matrix()
#     # Create dense tensor from numpy array
#     T = skt.dtensor(X)
#     # Decompose tensor using CP-ALS
#     P, fit, itr, exectimes = skt.cp_als(T, 2, init='random')
#     print (P.U)


# In[ ]:



