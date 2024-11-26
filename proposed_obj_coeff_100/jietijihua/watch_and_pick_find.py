
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 07:58:59 2022

@author: Administrator
"""

import numpy as np
import os
import shutil 
import os
import re
import math


import sys

import time

path=os.path.dirname(sys.argv[0])

fff = os.path.basename(path)

print('fff',fff)


duoshaodu = re.findall("\d+",fff)

print('duoshaodu',duoshaodu)


# du = int(duoshaodu[0]) 

# print('du',du)




seed_int = [1,2,3,4,5,6,7,8]


allguys_fake = []


for i in seed_int:
    
            
    allguys_fake.append( 's_' +   str(i) )



allguys = []


for n in allguys_fake:
        
    if os.path.exists(n):
        
        allguys.append(n)










def congwenjianjialistduquzuida(jihelist):
    
    jihejihe = []
    
    reward_jihe = []
    
    for xinwenjianjiaming in jihelist:
                
        wenjianming_1 = xinwenjianjiaming +'/' + 'AgentPPO/fuck_0/recorder0.6to0.3_bu0_case1/quan_recorder.npy'
                
        jihejihe.append(wenjianming_1)
    
    
    
    
    
    for iii in jihejihe:
                  
        if os.path.exists(iii):
            
            
            best_one_jihe = np.load(iii)
            
            the_last_bestone = best_one_jihe[:,1]
            
            reward_indicator = the_last_bestone[-1]
            
            reward_jihe.append(reward_indicator)
            
    
    print('reward_jihe',reward_jihe)
    
    if len(reward_jihe) == 0:
        
        return 0
    
    else:
        return sum(reward_jihe)*1.0/len(reward_jihe), max(reward_jihe), reward_jihe.index(max(reward_jihe))+1







while 1:

    
    nnn = 0
    
    dangezu = []
    
    meizuzuida = []
    
    zuidaweizhi = []
    
    pingjun = []
    
    for i in allguys:
        
        if os.path.exists(i+'\\6to3\\thebest.png'):
        
            shutil.copy(i+'\\6to3\\thebest.png',i+'.png')
        
    
    
    
        
    
    aaa,bbb,ccc = congwenjianjialistduquzuida(allguys)
    
    meizuzuida.append(bbb)
    
    pingjun.append(aaa)
    
    zuidaweizhi.append(ccc)
    
    dangezu = [] 

    zuidade_wenjianjia =  allguys[int(zuidaweizhi[0]-1)]
        
    print('meizuzuida',meizuzuida)
    
    print('zuida_suozaiwenjianjia', zuidade_wenjianjia)


    if os.path.exists(zuidade_wenjianjia+'\\6to3\\thebest.png'):
    
        shutil.copy(zuidade_wenjianjia+'\\6to3\\thebest.png','fuckisit'+'.png')
    
    
    print('zuidaweizhi',zuidaweizhi)
    
    
    max_todingzifu = '.'
    
    for hj in meizuzuida:
        
        max_todingzifu = max_todingzifu +'____' + str(round(hj,3))
        
    print('max_todingzifu',max_todingzifu)
    
    
    
    todingzifu = max_todingzifu

    



    jiluwenjian = zuidade_wenjianjia + '/6to3/the_best.npy'
    
    loadming_new_wenj = jiluwenjian
    
    initial = 0.3
    
    end = 0.6
    
    feigao_aim = end*0.4
    
    jilustates = np.load(loadming_new_wenj)
    
    

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
         
    
    
    
    zz = np.array(zhua)
    
    xx = np.array(xhua)
    
    cc = np.vstack((xx,zz)).T
    

    
    jianqu = zz - 0.4*end
    
    if initial < end:
        
        print('chaochu    ',max(jianqu))
        
        diyige = np.where(zz > 0.4*end)
    
    
    if initial > end:
        
        print('chaochu    ',min(jianqu))
        
        diyige = np.where(zz < 0.4*end)
    
    
    #diyige = np.where(zz > 0.4*end)
    
    
    # print('diyige',diyige)
    
    
    # print('len(diyige)',len(diyige[0]))
    
    if len(diyige[0])>0:
                                
        
        suoyinchang = diyige[0]
        
        youxiaochang = suoyinchang[0]
        
        qianjinjuli = zz[youxiaochang:]
        
        guo_cha =  abs(qianjinjuli - 0.4*end)
        
        print('abs_chaochu   ',max(guo_cha))
        
    else:
        
        print('abs_chaochu   ', 'meichaochu')
              


                                                          
    print('-------------------------------------------------')
    #print('avg_chaochu   ',np.mean(guo_cha))
    
    
    
    
    
    
    time.sleep(60)
    

































'''







while 1:
        
    shenme = re.findall(r"readallbest(.+?).py",name)
    
    print('shenme',shenme)
    
    guys = [shenme[0]] #[0,1,3,5,100]
    
    xuhao =  [1,2,3,4,5,6,7,8]
    
    zhiji_casexuhao_1 = []
    
    zhiji_casexuhao_2 = []
    
    zhiji_casexuhao_3 = []
    
    zhiji_casexuhao_4 = []
    
    wanchengbianhao = []
    
    weiwanchengbianhao = []
    
    allsamples_jihe = []
    
    allsamples_except_jihe = []
    
    for i in guys:
        
        for j in xuhao:
            
    
                xinwenjianjiaming =  'g' + i + 's-'+str(j)
                            
                wenjianming_1 = xinwenjianjiaming +'/' + '99' +'/' + 'bestlog.npy'
                
                # wenjianming_2 = xinwenjianjiaming +'/' + '2' +'/' + 'bestlog.npy'
                
                # wenjianming_3 = xinwenjianjiaming +'/' + '3' +'/' + 'bestlog.npy'
                
                # wenjianming_4 = xinwenjianjiaming +'/' + '4' +'/' + 'bestlog.npy'
                
                
    
                #for iii,jjj in zip([wenjianming_1,wenjianming_2,wenjianming_3,wenjianming_4],[zhiji_casexuhao_1,zhiji_casexuhao_2,zhiji_casexuhao_3,zhiji_casexuhao_4]) :
                    
                for iii,jjj in zip([wenjianming_1],[zhiji_casexuhao_1]) :
                    
                   
    
                    if os.path.exists(iii):
                        
                        best_one_jihe = np.load(iii)
                        
                        the_last_bestone = best_one_jihe[-1]
                        
                        reward_indicator = the_last_bestone[0]
                        
                        jjj.append(reward_indicator)
                        
                        allsamples_jihe.append(the_last_bestone[1:5])
                        
                        allsamples_except_jihe.append(the_last_bestone[-13:-1])
                        
                
                
                suanwanmei_1 = xinwenjianjiaming +'/' + '99' +'/' + 'over.npy'
                
                # suanwanmei_2 = xinwenjianjiaming +'/' + '2' +'/' + 'over.npy'
                
                # suanwanmei_3 = xinwenjianjiaming +'/' + '3' +'/' + 'over.npy'
                
                # suanwanmei_4 = xinwenjianjiaming +'/' + '4' +'/' + 'over.npy'
                
               # for kkk in [suanwanmei_1,suanwanmei_2,suanwanmei_3,suanwanmei_4]:
    
                for kkk in [suanwanmei_1]:
                                    
                    
                    if os.path.exists(kkk):
                        
                        wanchengbianhao.append(kkk)
                        
                    else:
                        
                        weiwanchengbianhao.append(kkk)                    
                        
                        
                    
                    
                    
    print('max(zhiji_casexuhao_1)',max(zhiji_casexuhao_1))
                    
                    
    #df.iloc[0,1] = max(zhiji_casexuhao_1)
                
    # df.iloc[0,2] = max(zhiji_casexuhao_2)
                
    # df.iloc[1,1] = max(zhiji_casexuhao_3)
            
    # df.iloc[1,2] = max(zhiji_casexuhao_4)             
                    
                    
                    
                    
    weizhi_1 = zhiji_casexuhao_1.index(max(zhiji_casexuhao_1))
    
    zuidaweizhichu = weizhi_1 + 1
    
    print('zuidazai_forcase1 ',zuidaweizhichu)         
    print('')
    print('zuidazai_forcase1_samples_except_sum', sum(allsamples_except_jihe[weizhi_1]))
    print('')
    
    print('zuidazai_forcase1_samples_sum', sum(allsamples_jihe[weizhi_1]))
    print('')
    print('zuidazai_forcase1_samples', allsamples_jihe[weizhi_1])
                    
    # weizhi_2 = zhiji_casexuhao_2.index(max(zhiji_casexuhao_2))
    
    # zuidaweizhichu = weizhi_2 + 1
    
    # print('zuidazai_forcase2 ',zuidaweizhichu)                 
                    
        
    # weizhi_3 = zhiji_casexuhao_3.index(max(zhiji_casexuhao_3))
    
    # zuidaweizhichu = weizhi_3 + 1
    
    # print('zuidazai_forcase3 ',zuidaweizhichu)     
    
        
    # weizhi_4 = zhiji_casexuhao_4.index(max(zhiji_casexuhao_4))
    
    # zuidaweizhichu = weizhi_4 + 1
    
    # print('zuidazai_forcase4 ',zuidaweizhichu) 
    
    
    
    
        
    print('weiwancheng',weiwanchengbianhao)
    print('')
    #print('fuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuck')
    print('')
    print('wancheng',wanchengbianhao)
    #print('')
    #print('fuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuckfuck')
    #print('')
    #print(df)
    
    print('zhiji_casexuhao_1',zhiji_casexuhao_1)
    print('samplesnum',len(zhiji_casexuhao_1))
    #df.to_excel('guinahou.xlsx', sheet_name='Sheet1')
    
    zifuchuan = 'more'+ '__'+ shenme[0] + '__' + str( round(max(zhiji_casexuhao_1),3))+'__' + str(round(sum(allsamples_except_jihe[weizhi_1]),3))
    
    print(zifuchuan)

    import json
    import requests
    def dingding_message(contennt):
        token="https://oapi.dingtalk.com/robot/send?access_token=9b8e277b9238acbe0d6a29b9a5d6e9d2adc467e51e10f78f1d369f785040c972"  #这里替换为你刚才复制的内容
        headers={'Content-Type':'application/json'}
        data={"msgtype":"text","text":{ "content":contennt}}
        requests.post(token,data=json.dumps(data),headers=headers)
    
    #dingding_message("ok.....包含刚才设置的自定义关键字")    

    dingding_message("ok."+zifuchuan)

    time.sleep(300)
    





'''










