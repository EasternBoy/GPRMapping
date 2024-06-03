import mRobot
import random
import torch
import gpytorch
import numpy as np

def update(robo, model):
   N = len(robo)
   
   neighbors = mRobot.find_nears(robo)
   
   for i in range(N):
       neighbors[i].append(i)
       for j in neighbors[i]:
            npos = torch.tensor([robo[j].cpos], dtype=torch.double)
            nmea = robo[j].cmea
            model[j].get_fantasy_model(npos, nmea)
   
   return 0

def rand_position(robo):
    x_min = robo[0].pBnd.x_min
    x_max = robo[0].pBnd.x_max
    y_min = robo[0].pBnd.y_min
    y_max = robo[0].pBnd.y_max
    
    s_max = robo[0].pBnd.s_max
    
    N = len(robo)
    R = robo[0].R
    r = robo[0].r
    
    flag = True
    
    while flag:
        s = [[random.uniform(robo[i].cpos[0] - s_max, robo[i].cpos[0] + s_max), random.uniform(robo[i].cpos[1] - s_max, robo[i].cpos[1] + s_max)] for i in range(N)]
        
        flag = False 
    
        for i in range(N):
            if s[i][0] < x_min or s[i][0] > x_max or s[i][1] < y_min or s[i][1] > y_max:
                flag = True
                
        # test = mRobot.check_nears(s, R, r, N)

        # if test:
        #     for i in range(N):
        #         robo[i].cpos = s[i]
        # else:
        #     flag = True
            

def proxADMM():
    
    return 0
