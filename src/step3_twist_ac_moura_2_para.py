##層間相互作用の計算で2パターン考える tetracene1 6分子 R3:tshaped R4:slipped-parallel
##2次元探索　cx cy
##step3 精密化　tetracene pentaceneパターン　a,θ,czを最適化 13分子計算
import os
os.environ['HOME'] ='/home/ohno'
import numpy as np
import pandas as pd
import time
import sys
from tqdm import tqdm
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

INTERACTION_PATH = os.path.join(os.environ['HOME'],'Working/interlayer_interaction/')
sys.path.append(INTERACTION_PATH)

from make_twist_ac_para import exec_gjf

from utils import get_E1


def main_process(args):
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    auto_csv_path = os.path.join(auto_dir,'step3_twist.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['cx','cy','cz','a','b','Rt','Rp','theta','A1','A2','phi1','E','E_i01','E_i02','E_ip1','E_ip2','E_ip3','E_ip4','E_it1','E_it2','E_it3','E_it4','machine_type','status','file_name'])##いじる
        df_E.to_csv(auto_csv_path,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args)
        time.sleep(1)

def listen(args):
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    num_nodes = args.num_nodes
    isTest = args.isTest
    ##isInterlayer =args.isInterlayer
    #### TODO
    fixed_param_keys = ['a','b','theta','A1','A2','phi1','Rt','Rp']
    opt_param_keys = ['cx','cy','cz']

    auto_csv = os.path.join(auto_dir,'step3_twist.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    maxnum_machine2 = 2#num_nodes/2 if num_nodes%2==0 else (num_nodes+1)/2##適宜変える
    
    for idx,row in zip(df_queue.index,df_queue.values):
        machine_type,file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E1(log_filepath)
        if len(E_list)!=10:##計算する層状の分子数
            continue
        else:##エネルギーの内訳全般
            len_queue-=1;machine_type_list.remove(machine_type)
            Ei01=float(E_list[0]);Eip1=float(E_list[1]);Eip2=float(E_list[2]);Eit1=float(E_list[3]);Eit2=float(E_list[4]);Eit3=float(E_list[5]);Eit4=float(E_list[6]);Ei02=float(E_list[7]);Eip3=float(E_list[8]);Eip4=float(E_list[9])##ここも計算する分子数に合わせて調整
            E = 2*((Ei01+Eip1+Eip2+Ei02+Eip3+Eip4)/2+Eit1+Eit2+Eit3+Eit4)##隣接20分子　2パターン
            #### TODO
            df_E.loc[idx, ['E_i01','E_ip1','E_ip2','E_it1','E_it2','E_it3','E_it4','E_i02','E_ip3','E_ip4','E','status']] = [Ei01,Eip1,Eip2,Eit1,Eit2,Eit3,Eit4,Ei02,Eip3,Eip4,E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes, fixed_param_keys, opt_param_keys)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                #### TODO
                file_name = exec_gjf(auto_dir, monomer_name,params_dict, machine_type,isInterlayer=True,isTest=isTest)##paramsdictとか
                df_newline = pd.Series({**params_dict,'E':0.,'E_i01':0.,'E_ip1':0.,'E_ip2':0.,'E_it1':0.,'E_it2':0.,'E_it3':0.,'E_it4':0.,'E_i02':0.,'E_ip3':0.,'E_ip4':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step3_twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step3_twist.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes, fixed_param_keys, opt_param_keys):
    """
    前提:
        step3_twist_init_params.csvとstep3_twist.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step3_twist_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step3_twist.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    #fixed_param_keys = ['a','b','theta','R3','R4']     opt_param_keys = ['cx','cy','cz']


    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        ### TODO
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)##TF cx cy czを出力
        if isDone:
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return {**fixed_params_dict,**opt_params_dict}
            else:
                continue

        else:
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
            if len(df_inprogress)>=1:
                continue
            return {**fixed_params_dict,**opt_params_dict}
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    cx_init_prev = init_params_dict['cx']; cz_init_prev = init_params_dict['cz']; cy_init_prev = init_params_dict['cy']
    rt = init_params_dict['Rt']; rp = init_params_dict['Rp']
    while True:
        E_list=[];params_list=[]
        for cx in [cx_init_prev]:
            for cy in [cy_init_prev]:
                for cz in [cz_init_prev]:  
                    cy = np.round(cy,1);cx = np.round(cx,1);cz= np.round(cz,1)
                    df_val_params = df_val[
                    (df_val['cy']==cy)&(df_val['cx']==cx)&(df_val['cz']==cz)&
                    (df_val['Rt']==rt)&(df_val['Rp']==rp)&
                    (df_val['status']=='Done')
                    ]
                    if len(df_val_params)==0:
                        return False,{'cx':cx,'cy':cy,'cz':cz}
                    params_list.append([cx,cy,cz]);E_list.append(df_val_params['E'].values[0])
        cx_init,cy_init,cz_init = params_list[np.argmin(np.array(E_list))]
        if cx_init==cx_init_prev and cy_init==cy_init_prev and cz_init==cz_init_prev:
            return True,{'cx':cx_init,'cy':cy_init, 'cz':cz_init}
        else:
            cx_init_prev=cx_init;cy_init_prev=cy_init;cz_init_prev=cz_init
    
def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    query = []
    for k, v in dict_filter.items():
        if type(v)==str:
            query.append('{} == "{}"'.format(k,v))
            df=df[df[k]==v]
        else:
            query.append('{} == {}'.format(k,v))
            df=df[df[k]==v]
    df['A1']=df['A1'].astype(float)##df.queryのバグ
    df['A2']=df['A2'].astype(float)##df.queryのバグ
    df['cy']=df['cy'].astype(float)##df.queryのバグ
    query0=' and '.join(query)
    df_filtered=df
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    ##parser.add_argument('--isInterlayer',action='store_true')##追加
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        #init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")