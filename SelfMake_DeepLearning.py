# coding:utf-8
import numpy as np
import pandas as pd
from numpy import linalg as la
import math
import matplotlib.pyplot as plt
import random
##########設定################
#層と各層のニューロンの数を指定
Units=np.array([4,100,3])
Layer=len(Units)
n=100000 #何回ミニバッチを使うか
n2=10 #ミニバッチの個数
f_type=0 #活性化関数の種類　0:シグモイド関数， 1:正規化線形関数
epsilon=0.001
ave=0
sigma=1
##############################

#活性化関数を定義
def f(x,i,j):#i:微分, j:種類
	if j==0:
		ans=1/(1+np.exp(-x))#ロジスティック曲線
		ans2=ans*(1-ans)#ロジスティック微分
	if j==1:
		ans2=np.array(x >=0,dtype=float)
		ans=ans2*x
	if i==0:
		return(ans)
	else:
		return(ans2)
    
#ソフトマックス関数を定義
def softmax(x):
    x1=np.exp(x)
    x2=np.sum(x1,axis=0)
    return(x1/x2)

#WとBの初期値を設定
W=range(Layer)  
#Rのvector的なものがないので．．．．
B=range(Layer)
for a in range(Layer):
    if a==0:
    	continue
    W[a]=np.random.normal(ave,sigma,(Units[a],Units[a-1]))
    B[a]=np.random.normal(ave,sigma,(Units[a],1))

#irisを呼びだす.
iris=pd.read_csv("iris.csv", header=None)
#numpyの形式にする．
iris_N=np.array(iris.ix[1:,1:])
for i in range(4):
	iris_N[:,i]=iris_N[:,i].astype(np.float64)

t=np.zeros((150,3))
iris_N=np.c_[iris_N,t]

#答えデータを行列で表す．
for i in range(len(iris_N)):
	if iris_N[i,4]=="setosa":
		iris_N[i,5]=iris_N[i,5]+1.0
	if iris_N[i,4]=="versicolor":
		iris_N[i,6]=iris_N[i,6]+1.0
	if iris_N[i,4]=="virginica":
		iris_N[i,7]=iris_N[i,7]+1.0

np.random.shuffle(iris_N)

test=np.array(iris_N[:50,:])
train=iris_N[50:,:]

for i in range(n):
	index = random.sample(range(100), n2)#抽出する添字を取得
	X=np.array(train[index,0:4].T,dtype=float)
	d=np.array(train[index,5:8].T,dtype=float)

	#U,Zを求める．
	U=range(Layer)
	Z=range(Layer)

	for a in range(Layer):
		if a==0:
			U[0]=X
			Z[0]=X
		else:
			U[a]=W[a].dot(Z[a-1])+B[a].dot( np.ones((X.shape)[1])[np.newaxis,:])
			if a !=Layer-1:
				Z[a]=f(U[a],0,f_type)
			else:
				Z[a]=softmax(U[a])
		
#デルタとWの偏微分を求める
	N=(X.shape)[1]*1.0 
	delta=range(Layer)
	Wd=range(Layer)
	Bd=range(Layer)
	for a in range(Layer-1,0,-1):
		if a==max(range(Layer)):
			delta[a]=Z[a]-d
		else:
			delta[a]=f(U[a],1,f_type)*((W[a+1].T).dot(delta[a+1]))
#W+Bの更新
        Wd[a]=(1/N)*((delta[a]).dot(Z[a-1].T))
        W[a]=W[a]-(epsilon*Wd[a])
        Bd[a]=(1/N)*(delta[a].dot(np.ones((X.shape)[1])[:,np.newaxis]))
        B[a]=B[a]-(epsilon*Bd[a])

############結果の表示############
#for i in range(1,Layer):
    #print("W",i)
    #print(pd.DataFrame(W[i]))
    #print("B",i)
    #print(pd.DataFrame(B[i]))
    
U=range(Layer)
Z=range(Layer)
X=np.array(test[:,0:4].T,dtype=float)
d=np.array(test[:,5:8].T,dtype=float)

for a in range(Layer):
    if a==0:
        U[0]=X
        Z[0]=X
    else:
        U[a]=W[a].dot(Z[a-1])+B[a].dot( np.ones((X.shape)[1])[np.newaxis,:])
        if a !=Layer-1:
            Z[a]=f(U[a],0,f_type)
        else:
            Z[a]=softmax(U[a])

res=np.r_[Z[Layer-1],d]
print(pd.DataFrame(res))
a=res[0:3,:]
b=res[3:6,:]
a2=np.array(np.where(np.max(a,axis=0)==a))
b2=np.array(np.where(b==1))
a3=pd.DataFrame(a2)
b3=pd.DataFrame(b2)
aix=np.array(((a3.T).sort(1)).index)
bix=np.array(((a3.T).sort(1)).index)
correct=np.array(np.where(a2[0,aix]==b2[0,bix]))
correct_len=correct.shape[1]
print("rcorrect rate is",(correct_len*1.)/(res.shape[1]))
print(a2[0,aix])
print(b2[0,bix])