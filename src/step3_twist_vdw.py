##エネルギー等計算の対象を確認　R3:t-shaped R4:slipped-parallel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Rod
from make_twist import get_monomer_xyzR
from scipy import signal
from scipy.ndimage.filters import maximum_filter

def get_c_vec_vdw(monomer_name,A1,A2,a_,b_,theta):#,name_csv
    
    i=np.zeros(3); a=np.array([a_,0,0]); b=np.array([0,b_,0]); t1=(a+b)/2;t2=(a-b)/2 ##ずらし方の定義
    monomer_array_i = get_monomer_xyzR(monomer_name,0.,0.,0.,A1,A2,theta)
    
    monomer_array_t = get_monomer_xyzR(monomer_name,0.,0.,0.,-A1,A2,-theta)
    
    arr_list1=[[i,'p'],[b,'p'],[-b,'p'],[a,'p'],[-a,'p'],[t1,'t'],[-t1,'t'],[t2,'t'],[-t2,'t']]##層内の分子の座標とp or t
    arr_list2=[[i,'t'],[b,'t'],[-b,'t'],[a,'t'],[-a,'t'],[t1,'p'],[-t1,'p'],[t2,'p'],[-t2,'p']]##層内の分子の座標とp or t
    Rb_list=[np.round(Rb,1) for Rb in np.linspace(-np.round(b_/2,1),np.round(b_/2,1),int(np.round(2*np.round(b_/2,1)/0.1))+1)]
    z_list=[];V_list=[]
    for Rb in tqdm(Rb_list):
        z_max1=0
        for R,arr in arr_list1:
            if arr=='t':
                monomer_array1=monomer_array_t
            elif arr=='p':
                monomer_array1=monomer_array_i
            for x1,y1,z1,R1 in monomer_array1:#層内
                x1,y1,z1=np.array([x1,y1,z1])+R
                for x2,y2,z2,R2 in monomer_array_i:#i0
                    x2+=0
                    y2+=Rb
                    z2+=0
                    z_sq=(R1+R2)**2-(x1-x2)**2-(y1-y2)**2
                    if z_sq<0:
                        z_clps=0.0
                    else:
                        z_clps=np.sqrt(z_sq)+z1-z2
                    z_max1=max(z_max1,z_clps)
        z_max2=0
        for R,arr in arr_list2:
            if arr=='t':
                monomer_array1=monomer_array_t
            elif arr=='p':
                monomer_array1=monomer_array_i
            for x1,y1,z1,R1 in monomer_array1:#層内
                x1,y1,z1=np.array([x1,y1,z1])+R
                for x2,y2,z2,R2 in monomer_array_t:#i0
                    x2+=0
                    y2+=Rb
                    z2+=0
                    z_sq=(R1+R2)**2-(x1-x2)**2-(y1-y2)**2
                    if z_sq<0:
                        z_clps=0.0
                    else:
                        z_clps=np.sqrt(z_sq)+z1-z2
                    z_max2=max(z_max2,z_clps)
        z_max=max(z_max1,z_max2)
        z_list.append(z_max)
        V_list.append(a_*b_*z_max)
    return np.array([0,Rb_list[np.argmin(V_list)],min(z_list)])##ここでは層間距離を計算している

def detect_peaks(image, filter_size):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))
    return detected_peaks

    

