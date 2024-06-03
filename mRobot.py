import torch
import random
import mGP
import numpy as np

class polyBound:
    s_max = 0.
    x_min = 0.
    x_max = 0.
    y_min = 0.
    y_max = 0.
    
    def __init__(self, s_max, x_min, x_max, y_min, y_max):
        self.s_max = s_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    

class robot:
    id = 0
    T = 0. # Sampling time
    H = 0  # Horizon length
    R = 0.
    r = 0.
    σ = 0.
    
    
    cpos = None
    cmea = None
    pBnd = None
    
    
    β   = None
    σκ2 = None
    ϕℓ2 = None
    σω2 = None
    iCθ = None
    
    def __init__(self, id, T, H, R, r, σ, pBnd, x0):
        self.id = id
        self.T = T
        self.H = H
        self.R = R
        self.r = R
        self.σ = σ
        
        self.pBnd = pBnd
        self.cpos = x0
        
        
def init_position(pBnd, R, r, N):
    x_min = pBnd.x_min
    x_max = pBnd.x_max
    y_min = pBnd.y_min
    y_max = pBnd.y_max
    
    while True:
        s = [[random.uniform(x_min+r, x_max-r), random.uniform(y_min+r, y_max-r)] for i in range(N)]
        if check_nears(s, R, r, N):
            return s
    
def check_nears(s, R, r, N):
    nearB = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if j != i:
                if (s[i][0] - s[j][0])**2 + (s[i][1] - s[j][1])**2 < 4*r**2:
                    return False
                if (s[i][0] - s[j][0])**2 + (s[i][1] - s[j][1])**2 < R**2:
                    nearB[i].append(j)
    for i in range(N):
        if len(nearB[i]) <= 1:
            return False
    
    return SecEig(nearB) > 1e-5

def SecEig(nearB):
    N = len(nearB)
    graph =  [[0]*N for _ in range(N)]
    
    for i in range(N):
        for j in nearB[i]:
            graph[i][j] = -1
            graph[i][i] += 1
    
    value, _ = np.linalg.eig(graph)
    value.sort()
    return value[1]


def find_nears(robo):
    N = len(robo)
    R = robo[0].R
    nearB = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if j != i:
                if (robo[0].cpos[0] - robo[0].cpos[0])**2 + (robo[0].cpos[1] - robo[0].cpos[1])**2 < R**2:
                    nearB[i].append(j)
    return nearB
