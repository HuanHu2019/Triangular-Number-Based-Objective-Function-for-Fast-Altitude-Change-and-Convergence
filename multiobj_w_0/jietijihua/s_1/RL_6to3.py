# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:43:16 2024

@author: ha
"""
import numpy as np
#import copy
from aeropy_fix import jiayou,chazhi,sampling,zhaoweizhi,runaero,Math_integrate,jixu
import os
import time
import torch
import numpy as np
import numpy.random as rd
import math
from copy import deepcopy
from agent import *
from run import *
from numpy import cos,sin,tan,sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

import matplotlib

import re
 
import sys
sys.path.append('../..')  # 添加上一级的上一级路径


tolerance = 0.002



rising_man_ori = [1]

for i in range(23):
    
    i = i + 1

    last_one = rising_man_ori[-1] 
    
    rising_man_ori.append(last_one + i)    ## 生成了26
    

rising_man = []
for x in rising_man_ori:
    rising_man.append(1)
print(rising_man)



rising_man.reverse()


maxmax = rising_man[0]+24


def extract_decimals(string):
    
    pattern =  r'\d+\.\d+' 
    
    decimals = re.findall(pattern,string) 
    
    return decimals 


path2 =os.path.abspath(os.path.join(os.getcwd(), "../.."))

tt  = os.path.basename(path2)

print('tt',tt)

s = extract_decimals(tt)


if len(s):
    
    

    print('s',s)
    
    sec_term_factor = float(s[0])
    
    print('sec_term_factor',sec_term_factor)

    
    
else:
    
    print('s',s)
        
    s = re.findall("\d+",tt)[0]
    
    sec_term_factor = float(s)

    print('sec_term_factor',sec_term_factor)





quanbu = True

gao_n = 0

gaocha_1 = 0.1

gaocha_2 = 0.1

gaocha_3 = 0.1

gaocha_4 = 0.1



initial = 0.6
end = 0.3

hua_x = 5
shijianbu = 0.02


zongbushu = 25

Tzhong = 2.9430000000000005*0.6/0.25
flap_max = 9.99999
jieshou = 0.0024

lhou=2*0.4

lqian=1.5*0.4

gabove=0.31*0.4


endheight = end * 0.4






ji=np.load('../../../datacollect.npy')
vmin=5
vmax=18
lenthsudu=100
suduzu=np.linspace(vmin,vmax,lenthsudu)
lentheta=100
thetazu=np.linspace(-11.5,5.5,lentheta)
hmax=0.4*1.31;  ###相当于1的相对飞高
hmin=0.4*0.41;  ###相当于0.1的相对飞高
lenh=100
hzu=np.linspace(hmin,hmax,lenh)
lenflap=100
flapzu=np.linspace(-10,10,lenflap)







allji = [0.3,0.6,0.9]

v_0_ji = v_aim_ji = [8.59920292,8.879512099,9.124996488]

Thrust_0_ji = Thrust_end_ji = [1.277167533,1.54278313,1.615092432]

theta_0_ji = theta_end_ji = [1.805507789,1.77565838,1.773445899]

h_0_ji = h_aim_ji = [0.243938438,0.363940457,0.483940605]

flap_0_ji = flap_end_ji = [-2.622866624,-2.303687751,-2.093826378]

feigao_0_ji = feigao_aim_ji =[0.3*0.4,0.6*0.4,0.9*0.4]





for i in  range(3):
    
    if int(10*end) == int(10*allji[i]) :
        
        
        v_aim = v_aim_ji[i]
        
        Thrust_end = Thrust_end_ji[i]
        
        feigao_aim = feigao_aim_ji[i]
        
        theta_end = theta_end_ji[i]
        
        h_aim = h_aim_ji[i]
        
        flap_end = flap_end_ji[i]

for i in  range(3):
    
    #print('i',i)
    
    if int(10*initial)==int(10*allji[i])  :
        
        v_0 = v_0_ji[i]
        
        Thrust_0 = Thrust_0_ji[i]
        
        feigao_0 = feigao_0_ji[i]
        
        theta_0 = theta_0_ji[i]
        
        #print('theta_0',theta_0)
        
        h_0 = h_0_ji[i]
        
        flap_0 = flap_0_ji[i]



i = int(initial*10)

j = int(end*10)


if (i==3 and j==9 ) or (i==9 and j==3):
    
    jieshoubu = 10+zongbushu-2 # zongbushu-2
    
else:
    
    jieshoubu = zongbushu-2

gamma = 1-1/(jieshoubu+2)



alpha_0 = theta_0

NavDDTheta= np.deg2rad(alpha_0)
NavDDPsi=0
NavDDPhi=0
 
MatDDC_g2b = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
MatDDC_g2b[0,0] = np.cos(NavDDTheta)*np.cos(NavDDPsi)
MatDDC_g2b[0,1] = np.cos(NavDDTheta)*np.sin(NavDDPsi)
MatDDC_g2b[0,2] = -np.sin(NavDDTheta)
MatDDC_g2b[1,0] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) - np.cos(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[1,1] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) + np.cos(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[1,2] = np.sin(NavDDPhi)*np.cos(NavDDTheta)
MatDDC_g2b[2,0] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) + np.sin(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[2,1] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) - np.sin(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[2,2] = np.cos(NavDDPhi)*np.cos(NavDDTheta)

Vx0 = v_0

Vz0 = 0

EarthDDG = 9.81

z0 = - h_0

transform_sudu =np.dot(MatDDC_g2b,np.array([Vx0,0,Vz0]).T)

transform_jiasudu=np.dot(MatDDC_g2b,np.array([0,0,EarthDDG]).T) 

state_0 = np.array([alpha_0,0,transform_sudu[0],transform_jiasudu[0],transform_sudu[2],transform_jiasudu[2],0,0,Vx0,Vz0,0,z0])  ###朝向下为正方向



 


class Fuckingfly:
    
    def __init__(self,initia_state_ori, Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu , gamma):
                
        self.state_0 = initia_state_ori[[0,2,3,4,5,6,7,10,11]]  ##输出使用
 
        self.initia_state_ori = initia_state_ori  ##
        
        self.state_ori = initia_state_ori
        
        self.max_step = int((hua_x)/(5*shijianbu))
        
        self.env_name = 'fuck'
        
        self.state_dim = 9
        
        self.action_dim=2
                
        self.if_discrete=False
        
        self.target_return=100000
        
        self.Tzhong = Tzhong
        self.flap_max = flap_max
        self.initial = initial
        self.end = end
        self.hua_x = hua_x
        self.shijianbu = shijianbu
        
        self.jilustates = np.append(initia_state_ori,[0,0,0,0,0,0,0,0,0,0,0,0])
        
        self.shineng_last = 0
        
        self.gamma = gamma
        
        self.jietidijibu = 0
        
        self.reward_total = 0
        
        
    def reset(self,casenum=1):
        
        self.state_ori = self.initia_state_ori  # 
        
        state_0_yuan = self.state_0
        
        
        self.jilustates = np.append(self.initia_state_ori,[0,0,0,0,0,0,0,0,0,0,0,0])
        
        self.shineng_last = 0
        
        self.jietidijibu = 0
        
        self.reward_total = 0
        
        self.Reward_for_settling_time = 0
        
        self.record_flying_height = []
        
        
        self.rising_done_or_not = False
        
        self.rising_done_before = True
                
        self.rising_done_after = False
        
        
        
        
      
        return state_0_yuan.astype(np.float32)
    
    def step(self, action):
        
        state_next_ori, reward_bu, done, fuck_fu =envstep(self.state_ori, action, self.Tzhong,    self.flap_max,    self.initial,    self.end,    self.hua_x,   self.shijianbu,   self.jilustates)
        
        self.state_ori = state_next_ori
        
        indexyao = [0,2,3,4,5,6,7,10,11]
        
        state_next_yuan = state_next_ori[indexyao]
        
        self.record_flying_height.append(fuck_fu[-1] )
        
        self.jilustates = np.vstack((self.jilustates, fuck_fu ))
        
        self.jietidijibu = self.jietidijibu + 1
        
   
        
        
        
        if done == True:
            
            if self.jietidijibu == 25:
                
                reward = 0
                
            else:
                    
                reward = -maxmax
                    
                    
            
            
        else:
                        
            shineng_now,xishu,xishu2,reward_pan,reward_pan_sec,reward_third,reward_forth = shinengjisuan(self.initial,    self.end ,   fuck_fu[-1], self.jietidijibu, self.record_flying_height)
                   
            #reward = (xishu*shineng_now - 0*self.shineng_last)*xishu2 + reward_bu + reward_pan_sec + reward_pan + reward_third + reward_forth
            
            reward = shineng_now
            
            self.shineng_last = shineng_now
            
            self.Reward_for_settling_time = 0
            
            
        #reward_together = reward + sec_term_factor * self.Reward_for_settling_time
        
            
        #self.reward_total = self.reward_total + reward_together 

        
            
        return state_next_yuan.astype(np.float32) , reward , done , {}
    
    def quanrender(self,casenum=1):
        
        quanhuatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.end, self.hua_x, self.shijianbu, self.reward_total)
        
    def render(self,casenum=1):
        
        huatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.end, self.hua_x, self.shijianbu, self.reward_total)
        
    def jisuanshangxia(self):
    
        chasum =  shangxiajiaocuo(self.jilustates)
        
        return chasum        
        
        
        
def find_first_smaller_element(arr, target):
    for i, element in enumerate(arr):
        if element <= target:
            return i
    return -1

def shinengjisuan(initial,end,feigao,jietidijibu,record_flying_height):
        
    
        fenmu = 0.4*abs(initial-end)
            
        juli = abs (feigao-feigao_aim )
        
        #dangqianbudereward = rightzhongreward[jietidijibu-1]

                
        result  = find_first_smaller_element(record_flying_height, 0.4*end)
        
        if result == -1:
            
            cost_n = 0
            
        else:
            
            result = result + 1
            
            if result <= jietidijibu:
                
                cost_n =  1 - np.clip(juli/fenmu,0,1) 
                
                if result == jietidijibu:
                    
                    cost_n = cost_n +  rising_man[ jietidijibu - (24 + 1) ]   # zongbushu = 25
        
        
        return cost_n*1,1,1,0,0,0,0
    
    
    
# def shinengjisuan(initial,end,feigao):
        
    
#         fenmu = 0.4*abs(initial-end)
            
#         juli = abs (feigao-feigao_aim )
        
#         # if  (1 - np.clip(juli/fenmu,0,2))>1:
            
#         #     cost_n = - (1 - np.clip(juli/fenmu,0,2))**2
                
#         # if  (1 - np.clip(juli/fenmu,0,2))<=1:
            
#         #     cost_n = (1 - np.clip(juli/fenmu,0,2))**2    
                
#         #     # if  juli <= jieshou:
                
#         #         # cost_n = 1 - np.clip(juli/fenmu,0,2)
                
#         cost_n =  1 - np.clip(juli/fenmu,0,1) 
                    
#         return cost_n*1,1,1,0,0,0,0          



def jisuan_settling_time_reward(initial,end,record_flying_height):
    
        settling_time_reward = 0     
        
        #print('record_flying_height',record_flying_height)
        
        if len(record_flying_height) :
        
            flying_height_reverse = record_flying_height[: :-1]      

            #print('flying_height_reverse',flying_height_reverse)                 
            
            for feigao in flying_height_reverse:
                        
                fenmu = 0.4*abs(initial-end)
                    
                juli = abs (feigao-feigao_aim )
                    
                if juli <= fenmu*0.5*tolerance:
                    
                    settling_time_reward = settling_time_reward + 1 
                    
                if juli > fenmu*0.5*tolerance:
                    
                    break
               
        return settling_time_reward
                    




            
def shangxiajiaocuo(jilustates):
    
        xzuobiao = jilustates[:,10]
        
        zzuobiao = -jilustates[:,11]
        
        jiao = jilustates[:,0]
           
        xhua, zhua =[] , [] 
        
        gabove = 0.31*0.4
        
        for a,b,c in zip(xzuobiao,zzuobiao,jiao):
             
             Theta_new = c
             
             Ggaodu = b
             
             Xgaodu = a
        
             xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
             
             zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
             
             xhua.append(xhua_n)
             
             zhua.append(zhua_n)
        
        zhua = np.array(zhua)
        
        chasum = 100
        
        if end > initial:
        
            diyige = np.where(zhua > endheight)
            
            suoyinchang = diyige[0]
            
            if len(suoyinchang) > 0:

                suoyindian = suoyinchang[0]
                      
                qianjinjuli = zhua[suoyindian:]
                
                cha = np.array(qianjinjuli) - endheight
                
                chasum = np.sum(np.abs(cha))

         
        
        
        if end < initial:
        
            diyige = np.where(zhua < endheight)
            
            suoyinchang = diyige[0]
            
            if len(suoyinchang) > 0:
                
                suoyindian = suoyinchang[0]
                            
                qianjinjuli = zhua[suoyindian:]
                
                cha = np.array(qianjinjuli) - endheight
                
                chasum = np.sum(np.abs(cha))

        return chasum            
         
 

 
  

def envstep(statearg,actionarg, Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu,  feixingjilu  ):

    
    done=False
    
    #np.set_printoptions(precision=2)
    # print("a:%.2f"%(a))
    #print("statearg:",statearg)
   
    flap,Thrust = actionarg[0]*flap_max, (actionarg[1]+1) * Tzhong/2
    
    # actionarg_2D = actionarg.reshape(1,-1)
    
    # actionarg_temp = acgui.transform(actionarg_2D)
    
    # actionarg_zhen = actionarg_temp[0]

    # flap,Thrust = actionarg_zhen[0], actionarg_zhen[1]
    
    Theta = statearg[0]
    Thetadot = statearg[1]
    u=statearg[2]
    udot=statearg[3]
    w=statearg[4]
    wdot=statearg[5]
    q=statearg[6]
    qdot=statearg[7]
    Vx=statearg[8]
    Vz=statearg[9]  ##朝下为正
    x=statearg[10]
    z=statearg[11]
    
    Ggaoduold = -z
    
    feigaoold=Ggaoduold-gabove*math.cos(Theta*math.pi/180)
    
    
    cost_1 = 0
    
    Thetasuan= np.arctan(Vz/Vx)+Theta 
    
    if Thetasuan+flap>10 or Thetasuan+flap<-10:
        
       # print('fuck Thetasuan+flap')
        
        done = True
                
        cost_1 = -np.clip((abs(Thetasuan+flap)-10)/10,0,1)
        
        next_state = statearg
        
        feigao = feigaoold
                
        cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 2 + 6 )/6 
        
        return next_state, cost, done, np.append(next_state, [cost, cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,flap,Thrust, feigao ])


    #  signal=engine.flyout(matlab.double([Thetasuan]),
     # matlab.double([Vx]),matlab.double([-1*z]),matlab.double([flap]),matlab.double([Thrust]),nargout=3)
    signal_ori=chazhi(Vx,Thetasuan,-1*z,flap,suduzu,thetazu,hzu,flapzu,ji)
    signal_sec=jiayou(Theta,-1*z,Thrust)
    signal=signal_ori+signal_sec
     
    Fx=signal[0]
    Fz=-1*signal[1]
    My=signal[2]
     
    Fy=0.  
    Mx=0.  
    Mz=0.
    Phi,Phidot=0.,0.   
    Psi,Psidot=0.,0.    
    v,vdot=0.,0.         
    p,pdot=0.,0.  
    r,rdot=0.,0.     
    Vy=0.   
    y=0.    
         
    outcome=runaero(Fx, Fy, Fz,Mx,My,Mz, \
             Phi,Theta,Psi,Phidot,Thetadot,Psidot, \
             u,v,w,udot,vdot,wdot, \
             p,q,r,pdot,qdot,rdot, \
             Vx,Vy,Vz,  \
             x,y,z,shijianbu,20,-20)
 
    Theta_new=outcome[1]
    Thetadot_new=outcome[4]
    u_new=outcome[6]
    udot_new=outcome[9]
    w_new=outcome[8]
    wdot_new=outcome[11]
    q_new=outcome[13]
    qdot_new=outcome[16]
    Vx_new=outcome[18]
    Vz_new=outcome[20]
    x_new=outcome[21]
    z_new=outcome[23]
          
     
    Ggaodu= -1*z_new
     
    next_state=np.array([Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new])
    
    feigao=Ggaodu-gabove*math.cos(Theta_new*math.pi/180)
    
    Thetasuan = np.arctan(Vz_new/Vx_new) + Theta_new 
      
    # cost=1*(feigao-dinggao)**2+0.01*Theta**2+0.01*Thetadot**2+0.01*w**2+0.01*wdot**2+0.01*q**2+0.01*qdot**2+0.01*Vz**2+0.1*(xzhongdian-x)
     
     #cost=1*(feigao-dinggao)**2+0.5*(xzhongdian-x)
     
     # if feigao >= dinggao:
         
     #     cost= (hmax-gabove-feigao)*10 + np.sqrt(3-(Vz-0)**2-(feigao-dinggao)**2)  # dinggao 0.51 *0.4
         
     # else:
         
     #     cost = (feigao-0.1*0.4)*10 + np.sqrt(3-(Vz-0)**2-(feigao-dinggao)**2) # zuidigao 0.41*0.4
             



     # if feigao >= dinggao:
         
     #     cost=  (hmax-gabove-feigao)*10 + np.sqrt(5-(Vz-0)**2)  # dinggao 0.51 *0.4
         
     # else:
         
     #     cost = (feigao-0.1*0.4)*10 + np.sqrt(5-(Vz-0)**2) # zuidigao 0.41*0.4
             
     


     # if feigao >= dinggao:
         
     #     yixiang=np.clip((Vz-0)/0.5,0,1)
         
     #     erxiang=np.clip((feigao-dinggao)/(hmax-gabove-dinggao),0,1)
         
     #     cost=  5-yixiang-erxiang # dinggao 0.51 *0.4
         
     # else:

     #     yixiang=np.clip((Vz-0)/0.25,0,1)
         
     #     erxiang=np.clip((dinggao-feigao)/(dinggao-0.1*0.4),0,1)
         
     #     cost=  5-yixiang-erxiang # zuidigao 0.41*0.4



    cost_2 = 0

    if Thetasuan<-11.5:
        
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
            
         cost_2 = -np.clip( (np.abs(Thetasuan)-11.5)/10 , 0 , 1 )
         
         cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
         cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 2 + 6 )/6 

         return next_state, cost, done, np.append(next_state, [cost, cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,flap,Thrust, feigao ])

    if Thetasuan>5.5:
        
        done=True
         
        next_state = statearg
        
        feigao = feigaoold
                    
        cost_2 = -np.clip( (np.abs(Thetasuan)-5.5)/10 , 0 , 1 )
        
        cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 2 + 6 )/6 

        return next_state, cost, done, np.append(next_state, [cost, cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,flap,Thrust, feigao ])

        # return next_state, cost, done, [Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new,cost,cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,cost_n,flap,Thrust ]
         #print('\n')
         #print("Theta+flap:    %.2f"%(Theta_new+flap),"    Theta:    %.2f"%(Theta_new),"    di:   %.2f"%(di),"           x:   %.2f"%(x))
         #print('\n')
        # print("Thetasuan:    %.2f"%(Thetasuan))
        # print('\n')



    cost_3 =-1  ##############################无用

    # jiasu = (Tzhong/1.6)*shijianbu

    # if feigao<dinggao-0.4*0.1:
                 
    #       if Vx_new < Vx:
             
    #           cost_3 = -np.clip( (Vx - Vx_new)/jiasu , 0 , 1 )         

    # if feigao>dinggao+0.4*0.1:
                 
    #       if Vx_new > Vx:
             
    #           cost_3 =-np.clip( (Vx_new - Vx)/jiasu , 0 , 1 )
    


    # if jiaopan == 0:
            
    #      done = True
            
    #      next_state = state_0   
         
    #      cost_3 = -1
          
    #      cost_4,cost_5,cost_6,cost_7,cost_8,cost_n=-1,-1,-1,-1,-1,-1,-1
        
    #      cost = cost_1+cost_2+cost_3+cost_4+cost_5+cost_6+cost_7+ cost_8 + cost_n  

    #      return next_state, cost, done, [Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new,cost,cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,cost_n,flap,Thrust ]   


    # cost_n = 0
    
    # h0=0.5*0.4
      
    # h = abs(feigao - feigao_aim)
    
    # if h <= h0:
        
        # cost_n = ((h0-h)/h0)**2
        
    # if h > h0:
        
        # cost_n =  - np.clip( (h - h0 )/h0, 0 ,1) 
     
            
   
    cost_4 = -1  ##############################无用



    # if feigao_0 > feigao_aim :
        
    #     if  hmax-gabove > feigao > feigao_0:
            
    #         cost_n = - ((feigao-feigao_0)/(hmax-gabove-feigao_0))**2
            
    #     if  feigao_aim <=feigao <= feigao_0:
        
    #         cost_n = ( (feigao - feigao_0)/(feigao_0-feigao_aim) )**2
    
    #     if  hmin-gabove < feigao <  feigao_aim :
        
    #         cost_n =((feigao -  (hmin-gabove )  )/(feigao_aim - (hmin-gabove )))**2
            



    # if feigao_0 < feigao_aim :
        
    #     if  hmax-gabove > feigao > feigao_aim :
            
    #         cost_n =  ((feigao-(hmax-gabove))/(hmax-gabove-(feigao_aim)))**2
            
    #     if  feigao_0 <= feigao <=feigao_aim  :
        
    #         cost_n =   ((feigao - feigao_0)/(feigao_aim -feigao_0))**2
    
    #     if  hmin-gabove < feigao <  feigao_0:
        
    #         cost_n  =    - ( (feigao - feigao_0)/(feigao_0 - (hmin-gabove )) )**2 
    
    # jieshou = 0.4*0.3*0.02
    
    # kaishiping = 0.4*0.3*0.3
    
    # fenmu = 0.4*abs(initial-end)
    
    # juli = abs (feigao-feigao_aim )
    
    # if juli >= kaishiping:
     
        # cost_n = 1 - np.clip(juli/fenmu,0,2)
        
    # if kaishiping >  juli > jieshou:
        
        # cost_n = 1 - np.clip( kaishiping /fenmu,0,2)
        
    # if  juli <= jieshou:
        
        # cost_n = 1 - np.clip(juli/fenmu,0,2)

    
            




    # jieshou = 0.4*0.3*0.02
    
    # kaishiping = 0.4*0.3*0.3
    
    # fenmu = 0.4*abs(initial-end)
    
    # juli = abs (feigao-feigao_aim )
    
     
    # cost_n = (1 - np.clip(juli/fenmu,0,2))**3
        

        
    # if  juli <= jieshou:
        
        # cost_n = 1 - np.clip(juli/fenmu,0,2)
        


    # if feigao_0 > feigao_aim :
        
        # if  hmax-gabove > feigao > feigao_0:
            
            # cost_n = - (feigao-feigao_0)/(hmax-gabove-feigao_0)
            
        # if  feigao_aim + jieshou < feigao < feigao_0:
        
            # cost_n = ( (feigao - (feigao_aim+jieshou))/(feigao_0-(feigao_aim+jieshou)) )**2
    
        # if  hmin-gabove < feigao <  feigao_aim - jieshou:
        
            # cost_n = ((feigao - (hmin-gabove))/((feigao_aim - jieshou) - (hmin-gabove )))**2
            
        # if   feigao_aim - jieshou <= feigao <=  feigao_aim + jieshou:
            
            # cost_n=1


    # if feigao_0 < feigao_aim :
        
        # if  hmax-gabove > feigao > feigao_aim+jieshou :
            
            # cost_n = 1- ((feigao-(feigao_aim+jieshou))/(hmax-gabove-(feigao_aim+jieshou)))**2
            
        # if  feigao_0 < feigao < feigao_aim - jieshou :
        
            # cost_n = ((feigao - feigao_0)/((feigao_aim - jieshou)-feigao_0))**2
    
        # if  hmin-gabove < feigao <  feigao_0:
        
            # cost_n = (feigao - (hmin-gabove))/(feigao_0 - (hmin-gabove )) - 1
            
        # if   feigao_aim - jieshou <=  feigao <=  feigao_aim + jieshou:
            
            # cost_n=1

















    cost_5 = 0
    if Theta_new>0:
        di=Ggaodu-gabove*math.cos(Theta_new*math.pi/180)-lhou*math.sin(Theta_new*math.pi/180)
         
    if Theta_new<0:
        di=Ggaodu-gabove*math.cos(Theta_new*math.pi/180)+lqian*math.sin(Theta_new*math.pi/180)
         
    if Theta_new==0:
        di=Ggaodu-gabove
     
    if di < 0.1*0.4:
         #print('fuck di')
        done=True
        
        next_state = statearg
        
        feigao = feigaoold
        
        cost_5 =  -np.clip( (0.1*0.4-di)/0.4, 0 , 1 )

        
         

    cost_6 = 0
     
    if Ggaodu >= hmax:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_6 = -np.clip( (Ggaodu-hmax)/0.4, 0 , 1 )
         
          
    elif Ggaodu <= hmin:
             
             done = True
             
             next_state = statearg
             
             feigao = feigaoold
             
             cost_6 =  -np.clip( (hmin-Ggaodu)/0.4, 0 , 1 )
             






    cost_7 = 0
     
    if Vx_new >= vmax:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 =  -np.clip( (Vx_new-vmax)/(vmax-vmin), 0 , 1 )
         

    if vmin >= Vx_new:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 = -np.clip( (vmin-Vx_new)/(vmax-vmin), 0 , 1 )
         



    cost_8 = 0
    
    if done==False:
        
        if jixu(Vx_new,Thetasuan,Ggaodu,0,suduzu,thetazu,hzu,flapzu,ji)==False:
            
            done = True
            
            next_state = statearg
            
            feigao = feigaoold
            
            cost_8 = -1
    
    if done==True:

            cost_8 = -1    
            
            
            
            
    # feigaojilu = feixingjilu[-1]
    
    # jieshou = 0.4*0.3*0.02
    
    # xia = feigao_aim - jieshou
    
    # shang = feigao_aim + jieshou

    # guf = np.where((feigaojilu >= xia) & (feigaojilu <= shang))

    # allguy = guf[0]

    # hh= len(allguy)
    
    # if (hh == 0) and (x_new > hua_x):
        
    #     done = True
        
        
    
    # fir = allguy[0]
    
    # chaoguofir = len(feigaojilu) - (fir+1)

    # if chaoguofir >  20:
         
    #      done = True

    

    if feixingjilu.shape[0]==1:
        
        bushu = 0

    else:
        
        bushu = len(feixingjilu) - 1
    

    
    if x_new < -1 * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
    
    if x_new > (jieshoubu+2) * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
                
    
    

    if bushu >  jieshoubu :
         
            done = True
                  
            next_state = statearg
            
            feigao = feigaoold  






    cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 2 + 6 )/6 
    
    return next_state, cost, done, np.append(next_state, [cost, cost_1,cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8,flap,Thrust, feigao ])   ## 12 +12



fucknum=0

matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'






#np.set_printoptions(precision=3)

fucknum=0

def huatu(jilustates,  Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu, reward_total):
    
    #print('jilustates',jilustates)
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fucknum
    
    fucknum=fucknum+1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = initial_zifu + 'to' + end_zifu
    
    if not os.path.exists(pathh+'\\' + wenjianjia):
            
        os.mkdir(pathh+'\\' + wenjianjia )
    
    mingcheng = pathh+'\\'+ wenjianjia+ '\\' + 'thebest'+'.png'
    
    #data_save = os.path.join(pathh, '\\', wenjianjia, '\\', str(fucknum)+'.npy')
    #data_save =pathh+ '\\'+ wenjianjia+ '\\'+ str(fuckquannum)+'.npy'
    
    data_save_2 =pathh+ '\\'+ wenjianjia+ '\\'+ 'the_best'+'.npy'
    
    #show_save =pathh+ '\\'+ wenjianjia+ '\\show'+ str(fucknum)+'.npy'
    
    #np.save(data_save,jilustates)
    
    np.save(data_save_2,jilustates)
    
    
    
    fig=plt.figure(figsize=(18,9))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
    
    gabove = 0.31*0.4
    
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
         Theta_new = c
         
         Ggaodu = b
         
         Xgaodu = a
    
         xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
         zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
         xhua.append(xhua_n)
         
         zhua.append(zhua_n)
        
    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker=markerzu[atr[0][1]],alpha=0.8)
    
    # for a,b,c in zip(xzuobiao,zzuobiao,jiao):
        
        # axes1.arrow(a,b,RR * np.cos(np.pi*c/180),RR * np.sin(np.pi*c/180),lw=0.01, fc='k', ec='k',linestyle="-")
    
    axes1.plot([0,hua_x],[0.4*initial,0.4*initial],linestyle='--')
    #axes1.plot([0,hua_x],[0.4*end,0.4*end],linestyle='--')
    
    jieshou = 0.5*tolerance*abs(initial-end)*0.4
    
    axes1.plot([0,hua_x],[0.4*end+jieshou,0.4*end+jieshou],linestyle='--')
    axes1.plot([0,hua_x],[0.4*end-jieshou,0.4*end-jieshou],linestyle='--')
    
    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    #ax.set_ylabel(r'$h$',font2,rotation=0)
    
    #ding  =  [initial, end]
    
    #axes1.set_ylim([(-0.31*0.25+ 0.55)*0.4, (0.31*0.25+ 0.6)*0.4])
    
    #axes1.set_ylim([0.4*end-jieshou - 0.025*0.4, 0.4*end+jieshou + 0.0125*0.4])
    
    
    #axes1.set_xlim([0, hua_x-0.5])
    
    axes1.set_xlim([0, 5])

    axes1.set_ylim([0.07, 0.26])
    
    
    
    #plt.tight_layout()
    
    axes1.xaxis.labelpad = 0
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
    
    plt.title(reward_total, y=2.3)
    


    
    plt.savefig(mingcheng,dpi=300)
    
    plt.close()





fuckquannum = 0


def quanhuatu(jilustates,  Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu, reward_total):
    
    #print('jilustates',jilustates)
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fuckquannum
    
    fuckquannum=fuckquannum+1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = 'quanhua'+ initial_zifu + 'to' + end_zifu
    
    if not os.path.exists(pathh+'\\' + wenjianjia):
            
        os.mkdir(pathh+'\\' + wenjianjia )
    
    #mingcheng = pathh+'\\'+ wenjianjia+ '\\' + str(fuckquannum)+'.png'
    
    
    mingcheng = pathh+'\\'+ wenjianjia+ '\\' + 'quan'+'.png'
    
    #data_save = os.path.join(pathh, '\\', wenjianjia, '\\', str(fucknum)+'.npy')
    #data_save =pathh+ '\\'+ wenjianjia+ '\\'+ str(fuckquannum)+'.npy'
    
    data_save_2 =pathh+ '\\'+ wenjianjia+ '\\'+ 'quan_best'+'.npy'
    
    #show_save =pathh+ '\\'+ wenjianjia+ '\\show'+ str(fucknum)+'.npy'
    
    #np.save(data_save,jilustates)
    
    np.save(data_save_2,jilustates)
    
    
    
    fig=plt.figure(figsize=(18,9))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
    
    gabove = 0.31*0.4
    
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
         Theta_new = c
         
         Ggaodu = b
         
         Xgaodu = a
    
         xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
         zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
         xhua.append(xhua_n)
         
         zhua.append(zhua_n)
        
    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker=markerzu[atr[0][1]],alpha=0.8)
    
    # for a,b,c in zip(xzuobiao,zzuobiao,jiao):
        
        # axes1.arrow(a,b,RR * np.cos(np.pi*c/180),RR * np.sin(np.pi*c/180),lw=0.01, fc='k', ec='k',linestyle="-")
    
    axes1.plot([0,hua_x],[0.4*initial,0.4*initial],linestyle='--')
    #axes1.plot([0,hua_x],[0.4*end,0.4*end],linestyle='--')

    jieshou = 0.5*tolerance*abs(initial-end)*0.4
    
    axes1.plot([0,hua_x],[0.4*end+jieshou,0.4*end+jieshou],linestyle='--')
    axes1.plot([0,hua_x],[0.4*end-jieshou,0.4*end-jieshou],linestyle='--')    

    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    #ax.set_ylabel(r'$h$',font2,rotation=0)
    
    #ding  =  [initial, end]
    
    #axes1.set_ylim([(-0.31*0.25+ 0.55)*0.4, (0.31*0.25+ 0.6)*0.4])
    
    #axes1.set_ylim([(-0.31*0.25+ 0.55)*0.4, (0.31*0.25+ 0.6)*0.4])
    
    #axes1.set_ylim([0.4*end-jieshou - 0.025*0.4, 0.4*end+jieshou + 0.0125*0.4])
    
    
    #axes1.set_xlim([0, hua_x-0.5])
    
    axes1.set_xlim([0, 5])

    axes1.set_ylim([0.07, 0.26])
    
    
    #plt.tight_layout()
    
    axes1.xaxis.labelpad = 0
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
    

    plt.title(reward_total, y=2.3)
    
    plt.savefig(mingcheng,dpi=300)
    
    plt.close()











def demo3_custom_env_rl (    Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu  ):
    
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    #args.random_seed = 1943

    '''choose an DRL algorithm'''
    #from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True

    args.allstep_num = jieshoubu+2

    
    args.gamma = 1-1/(jieshoubu+2)
    args.env = Fuckingfly(state_0, Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu, args.gamma)
    args.env_eval = Fuckingfly(state_0, Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu, args.gamma)  
    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
      # break training after 'total_step > break_step'
    
   
    
    args.if_remove = False

    args.if_allow_break = False
    
    #args.max_memo = 2 ** 21
    #args.batch_size = 2 ** 9
    #args.repeat_times = 2 ** 1
    args.eval_gap = 0  # for Recorder
    # args.eval_times1 = 2 ** 1  # for Recorder
    # args.eval_times2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    #args.rollout_num = 4
    args.eval_times1 = 2 **0
    args.eval_times1 = 2 **0
    #args.if_per = True

    


    train_and_evaluate(args,initial,end)










    

if __name__ == '__main__':

    demo3_custom_env_rl (  Tzhong,    flap_max,    initial,    end,    hua_x,   shijianbu )

