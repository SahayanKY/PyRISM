import json
import itertools

import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag
from scipy.fft import dst


def fft3d_spsymm(r, dr, k, dk, f):
    """
    f: shape: (numgrid, *, *)
    """
    t_f = 4*np.pi/k[:,np.newaxis,np.newaxis] * dst(r[:,np.newaxis,np.newaxis] * f, type=1, axis=0) / 2 * dr
    return t_f

def ifft3d_spsymm(r, dr, k, dk, t_f):
    """
    t_f: shape: (numgrid, *, *)
    """
    f = 1/(2*np.pi**2 * r[:,np.newaxis,np.newaxis]) * dst(k[:,np.newaxis,np.newaxis] * t_f, type=1, axis=0) / 2 * dk
    return f


# インプット読み込み
inputFile = 'sampleinput.json'
jsonDict = json.load(open(inputFile, 'r'))


# 収束パラメータ
mixingParam = jsonDict['config']['mixingParam']
chargeUp = jsonDict['config']['chargeUp']
criteria = jsonDict['config']['converge']
maxIterNum = jsonDict['config']['maxiter']

# 温度情報
temperature = jsonDict['temperature'] # 温度: K
beta = 1 / (1.98720e-3 * temperature) # 逆温度: (kcal/mol)^-1

# グリッド情報読み込み
grid1DDict = jsonDict['discretize']['grid1D']
dr = grid1DDict['dr'] # 刻み幅: A
numgrid = grid1DDict['n']
dk = np.pi / dr / (numgrid+1) # 刻み幅: A^-1
r = (np.arange(numgrid) + 1) * dr # shape: (numgrid,) # 動径位置: A
k = (np.arange(numgrid) + 1) * dk # shape: (numgrid,) # 動径位置: A^-1

# 溶媒データ読み込み
solventList = jsonDict['solvent']
# 溶媒種数
M = len(solventList)
solvName = [solv['name'] for solv in solventList]
# サイトデータ読み込み
siteList = [solv['site'] for solv in solventList]
# サイト数
Ns = [len(l) for l in siteList] # shape: (M,) # 溶媒種毎のサイト数
totalN = sum(Ns)
siteName = [l[0] for l in siteList] # shape: (totalN,) # サイト名
sigma = np.array(itertools.chain.from_iterable([l[1] for l in siteList])).reshape(-1)  # shape: (totalN,) # LJ sigma_ii:   A
eps = np.array(itertools.chain.from_iterable([l[2] for l in siteList])).reshape(-1)    # shape: (totalN,) # LJ epsilon_ii: kcal/mol
z = np.array(itertools.chain.from_iterable([l[3] for l in siteList])).reshape(-1)      # shape: (totalN,) # サイト電荷:    e
xyz = np.array(itertools.chain.from_iterable([l[4:] for l in siteList])).reshape(-1,3) # shape: (totalN, 3) # サイト座標:  A

# 単位行列
I = np.diag(np.ones(totalN)) # shape: (totalN, totalN)

# 数密度: A^-3
rhos = [solv['rho'] for solv in solventList] # shape: (M,)
P = np.diag([rhos[i] for i in range(M) for j in range(Ns[i])]) # shape: (totalN,totalN)

# 分子内サイト間距離行列: A
L = distance.cdist(xyz, xyz) # shape: (totalN, totalN) # 本来は異なる分子種間の距離は計算する意味がないが面倒なのでこのままで
# ブロック単位行列
bI = block_diag(*[np.ones([n,n]) for n in Ns]) # shape: (totalN, totalN) # delta_st
# 分子内相関行列(波数空間): [無次元]
t_W = bI * np.sinc(k[:,np.newaxis,np.newaxis] * L) # shape: (numgrid,totalN,totalN)

# サイト間短距離ポテンシャル行列: [無次元]: shape: (numgrid, totalN, totalN)
# Lorentz-Berthelot則
Sigma = (sigma[:,np.newaxis] + sigma) / 2           # shape: (totalN,totalN) # LJ sigma_ij
Eps = np.sqrt(eps[:,np.newaxis]*eps)                # shape: (totalN,totalN) # LJ eps_ij
__sigmar6 = (Sigma / r[:,np.newaxis,np.newaxis])**6 # shape: (numgrid, totalN, totalN)
Us = beta * 4 * Eps * (__sigmar6**2 - __sigmar6)
# サイト間長距離ポテンシャル行列: [無次元]: shape: (numgrid, totalN, totalN)
Ul = beta * 332.053 * z[:,np.newaxis] * z / r[:,np.newaxis,np.newaxis]

# サイト間長距離直接相関行列: shape: (numgrid, totalN, totalN)
# (実空間): [無次元]
Cl = Ul
# (波数空間): A^3
t_Cl = -beta * 332.053 * z[:,np.newaxis] * z * 4*np.pi / (k[:,np.newaxis,np.newaxis]**2)
# サイト間長距離全相関行列  : shape: (numgrid, totalN, totalN)
# (波数空間): A^3
t_Hl = t_W @ t_Cl @ t_W @ np.linalg.inv(I - P @ t_Cl @ t_W)
# (実空間): [無次元]: フーリエ逆変換
Hl = ifft3d_spsymm(r, dr, k, dk, t_Hl)

# Hypervertex
t_Omega = t_W + P@t_Hl
t_OmegaT = t_Omega.transpose(0,2,1)

# initialize
# 間接相関 Hs - Cs: shape: (numgrid, totalN, totalN)
Eta = np.zeros(Cl.shape)

numLoop = 0
while True:
    numLoop += 1
    if numLoop > maxIterNum:
        break

    # サイト間短距離直接相関行列: shape: (numgrid, totalN, totalN)
    Cs = np.exp(-Us+Eta) -Eta -1 # HNC closure
    # フーリエ変換
    t_Cs = fft3d_spsymm(r, dr, k, dk, Cs)

    # サイト間短距離全相関行列  : shape: (numgrid, totalN, totalN)
    # (波数空間): A^3
    t_Hs = t_OmegaT @ t_Cs @ t_Omega @ np.linalg.inv(I - P @ t_Cs @ t_Omega)
    # (実空間): [無次元]
    Hs = ifft3d_spsymm(r, dr, k, dk, t_Hs)

    # 間接相関更新
    newEta = Hs - Cs
    # 収束判定
    maxError = np.max(newEta - Eta)
    if maxError < criteria:
        break
    Eta = mixingParam * newEta + (1-mixingParam) * Eta
















