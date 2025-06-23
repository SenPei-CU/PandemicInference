import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import f
from scipy.stats import ttest_ind
import scipy.stats as st
import seaborn as sns
import networkx as nx
from scipy.stats import ttest_1samp
import time
import concurrent.futures
from multiprocessing import Pool, cpu_count
import traceback
import math
import warnings
import os

def get_slurm_job_id():
    job_id = os.environ.get('SLURM_JOB_ID')
    return job_id


#SIR Model
def SIR_simulation_predict(s,i,r,n,t,N,C,beta,L,D,gamma,fly,fly_time,edge_matrix):
    flag=0
    state=np.zeros([3,t,n])
    new_increase_I=np.zeros([t,n])
    MSA_pop=np.zeros([t,n])
    state[0,0,:]=s
    state[1,0,:]=i
    state[2,0,:]=r
    MSA_pop[0][:]=N
    C_rate=C/np.sum(C,axis=1).reshape(n,1)
    M=np.zeros([len(N),len(N)+1])
    i_commute_fly=np.zeros([2,n,n])
    for k in range(t-1):
        M[:,0:len(N)]=(fly[int(fly_time),:,:]*edge_matrix)/np.sum(C,axis=1).reshape(len(N),1)
        fly_time=fly_time+1
        state_Flow_Matrix_multinomial=np.zeros([3,len(N),len(N)])
        for u in range(len(N)):
           state_Flow_Matrix_multinomial[0,u,0:len(N)]=np.random.multinomial(state[0][k][u],M[u,:])[0:len(N)]
           state_Flow_Matrix_multinomial[1,u,0:len(N)]=np.random.multinomial(state[1][k][u],M[u,:])[0:len(N)]
           state_Flow_Matrix_multinomial[2,u,0:len(N)]=np.random.multinomial(state[2][k][u],M[u,:])[0:len(N)]
        sum_1 = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=0)
        sum_2 = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=1)
        sum_3 = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=0)
        sum_4 = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=1)
        sum_5 = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=0)
        sum_6 = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=1)
        sum_7_matrix = np.zeros([len(N),len(N)])
        S_rate=state[0][k].reshape(n,1)*C_rate
        delta_I=np.sum(beta*(state[1][k]/MSA_pop[k])*S_rate,axis=0)
        delta_I_nb=np.zeros(len(N))
        p=(1/np.sum(S_rate,axis=0))*S_rate
        nonzero_indices = np.nonzero(delta_I)
        choice = list(nonzero_indices[0])
        test_times=gamma*state[1,k,:]
        test_times[test_times==0]=0.000000001
        delta_I_nb=np.random.negative_binomial(test_times,test_times/(test_times+delta_I))
        for u in choice:
            sum_7_matrix[u,:]=np.random.multinomial(int(delta_I_nb[u]),p[:,u])
        sum_7=np.sum(sum_7_matrix,axis=0)
        r_to_s_Vector=np.random.poisson(lam= state[2,k,:] / L,size=len(N))
        i_to_r_Vector=np.random.poisson(lam= state[1,k,:] / D,size=len(N))
        state[0,k+1,:] = state[0,k,:] - sum_7 + r_to_s_Vector + (sum_1 - sum_2)
        state[1,k+1,:] = state[1,k,:] + sum_7 - i_to_r_Vector + (sum_3 - sum_4)
        state[2,k+1,:] = state[2,k,:] + i_to_r_Vector -  r_to_s_Vector + (sum_5 - sum_6)
        N = state[0,k+1,:]+state[1,k+1,:]+state[2,k+1,:]
        MSA_pop[k + 1][:] = N
        new_increase_I[k+1,:]=sum_7
        i_commute_fly[0,:,:]=i_commute_fly[0,:,:]+sum_7_matrix
        i_commute_fly[1,:,:]=i_commute_fly[1,:,:]+state_Flow_Matrix_multinomial[1,:,:]
        if np.min(state[:,k+1,:])<0:
            flag=1
            for i in range(len(N)):
                if np.min(state[:,k+1,i])<0:
                    sum=state[0,k+1,i]+state[1,k+1,i]+state[2,k+1,i]
                    for j in range(3):
                        if state[j][k+1][i]<0:
                            state[j][k+1][i]=1
                    weight=state[:,k+1,i]/sum
                    state[:,k+1,i]=sum*weight
                    for j in range(3):
                        state[j][k+1][i]=int(state[j][k+1][i])
  
    return state[0],state[1],state[2],MSA_pop,N,new_increase_I,flag,fly_time,i_commute_fly

def SIR_EAKF(s,i,r,n,N,C,beta,L,D,gamma,fly,fly_time,edge_matrix):
    flag=0
    state=np.zeros([3,n])
    new_increase_I=np.zeros(n)
    C_rate=C/np.sum(C,axis=1).reshape(n,1)
    M=np.zeros([n,n+1])
    for k in range(1):
        M[:,0:n]=(fly[int(fly_time),:,:]*edge_matrix)/np.sum(C,axis=1).reshape(n,1)
        state_Flow_Matrix_multinomial=np.zeros([3,n,n])
        for u in range(n):
           state_Flow_Matrix_multinomial[0,u,0:n]=np.random.multinomial(s[u],M[u,:])[0:n]
           state_Flow_Matrix_multinomial[1,u,0:n]=np.random.multinomial(i[u],M[u,:])[0:n]
           state_Flow_Matrix_multinomial[2,u,0:n]=np.random.multinomial(r[u],M[u,:])[0:n]
        sum_1 = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=0)
        sum_2 = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=1)
        sum_3 = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=0)
        sum_4 = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=1)
        sum_5 = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=0)
        sum_6 = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=1)
        sum_7_matrix = np.zeros([n,n])
        S_rate=s.reshape(n,1)*C_rate
        delta_I=np.sum(beta*(i/N)*S_rate,axis=0)
        delta_I_nb=np.zeros(n)
        p=(1/np.sum(S_rate,axis=0))*S_rate
        nonzero_indices = np.nonzero(delta_I)
        choice = list(nonzero_indices[0])
        test_times=gamma*i
        test_times[test_times==0]=0.000000001
        delta_I_nb=np.random.negative_binomial(test_times,test_times/(test_times+delta_I))
        for u in choice:
            sum_7_matrix[u,:]=np.random.multinomial(int(delta_I_nb[u]),p[:,u])
        sum_7=np.sum(sum_7_matrix,axis=0)
        r_to_s_Vector=np.random.poisson(r / L,size=n)
        i_to_r_Vector=np.random.poisson(i / D,size=n)
        state[0,:] = s - sum_7 + r_to_s_Vector + (sum_1 - sum_2)
        state[1,:] = i + sum_7 - i_to_r_Vector + (sum_3 - sum_4)
        state[2,:] = r + i_to_r_Vector -  r_to_s_Vector + (sum_5 - sum_6)
        N = state[0,:]+state[1,:]+state[2,:]
        new_increase_I=sum_7
        if np.min(state)<0:
            flag=1
            for i in range(len(N)):
                if np.min(state[:,i])<0:
                    sum=state[0,i]+state[1,i]+state[2,i]
                    for j in range(3):
                        if state[j][i]<0:
                            state[j][i]=1
                    weight=state[:,i]/sum
                    state[:,i]=sum*weight
                    for j in range(3):
                        state[j][i]=int(state[j][i])
  
    return state,N,new_increase_I,flag,sum_7_matrix,state_Flow_Matrix_multinomial[1,:,:]

#EAKF
def Getstandard(prior_sam_state,num_ens,uhf_id,N_all):
    post_sam_state=np.zeros((num_ens,3))
    for i in range(num_ens):
        for j in range(3):
            if prior_sam_state[i,j] < 0:
                prior_sam_state[i,j] = 0
        sum=np.sum(prior_sam_state[i])
        weight=prior_sam_state[i]/sum
        post_sam_state[i]=N_all[i,uhf_id]*weight
    return post_sam_state

def checkBound(prior_data,flag,uhf_id,num_ens,N_all):
    post_data=prior_data.copy()
    if flag==0:
        if post_data[prior_data>N_all[:,uhf_id]].size>0 or post_data[prior_data<0].size>0:
            for i in range(num_ens):
                if prior_data[i]>N_all[i,uhf_id]:
                    post_data[i]=0.9*N_all[i,uhf_id]
                if prior_data[i]<0:
                    post_data[i]=random.uniform(0.1,0.3)*np.mean(prior_data)

    else:    
        if post_data[prior_data>N_all[:,uhf_id]].size>0 or post_data[prior_data<0].size>0:
             for i in range(num_ens):
                if prior_data[i]>N_all[i,uhf_id]:
                    post_data[i]=0.5*N_all[i,uhf_id]
                if prior_data[i]<0:
                    post_data[i]=random.uniform(0.1,0.3)*np.mean(prior_data)

    return post_data

def Get_modify(prior_state,ens_id,N_all):
    post_state=prior_state.copy()
    for l in range(len(ens_id)):
        i=int(ens_id[l])
        for k in range (9):
            for j in range(3):
                if prior_state[i][j][k]<0:
                    post_state[i][j][k]=random.uniform(0.1,0.3)*np.mean(prior_state[:,j,k])
                if prior_state[i][j][k]>N_all[i,k]:
                    post_state[i][j][k]=random.uniform(0.8,0.9)*N_all[i,k]
            
            sum=np.sum(post_state[i,:,k])
            weight=post_state[i,:,k]/sum
            post_state[i,:,k]=N_all[i,k]*weight
    
    return post_state

def checkParaBound(prior_data, flag, p0, num_ens):
    post_data = prior_data.copy()
    if flag == 0:
        if post_data[prior_data > 0.45].size > 0 or post_data[prior_data < 0.3].size > 0:
            for i in range(num_ens):
                if prior_data[i] > 0.45:
                    post_data[i] = random.uniform(0.4, 0.45)
                if prior_data[i] < 0.3:
                    post_data[i] =  random.uniform(0.3, 0.35)

    else:
        if post_data[prior_data > 2 * p0[flag]].size > 0 or post_data[prior_data < 2].size > 0:
            for i in range(num_ens):
                if prior_data[i] < 2:
                    post_data[i] =  np.mean(prior_data)
                if prior_data[i] > 2 * p0[flag]:
                    post_data[i] = min(random.uniform(1.5, 1.6) * np.mean(prior_data),
                                       random.uniform(1.5, 1.6) * p0[flag])
    return post_data

def train_EAKF(MSA_real_case,T,obs_power,sam_0,para_0,num_ens,wnum,p0,N_all,C,fly,edge_matrix,real_time,D_sam,L_sam):
    MSA_observe=[]
    all_numens_i_commute_fly=np.zeros([2,len(N_all),len(N_all)])
    for id in range(len(C)):
        if np.sum(MSA_real_case[:,id])>0:
            MSA_observe.append(id)
    MSA_sam_record=np.zeros((T,num_ens,3,len(N_all)))
    MSA_sam_record_1=np.zeros((T,num_ens,1,len(N_all)))
    para_record=np.zeros((wnum,num_ens,len(N_all)))
    MSA_sam_record[0]=sam_0
    para_record[0]=para_0
    post_case=np.zeros((wnum-1,len(N_all),num_ens))
    N_All=np.zeros([T,num_ens,len(N_all)])
    for i in range(num_ens):
        N_All[0][i]=N_all
    for week in range(wnum-1):
        truth = MSA_real_case[week]
        oevbase = (100) ** 2  
        obs_var = np.array(oevbase + truth ** obs_power / 4)  
        sim_case = np.zeros((7,num_ens,len(N_all)))
        index_t = 0
        for t in range(7):
            index_t = 7 * week + t
            # EAKF
            for j in range(num_ens):  
                n=len(N_all)
                beta_id=[k for k in range(n)]
                beta=para_record[week,j,beta_id]
                D=D_sam[j]
                gamma=2.36
                L=L_sam[j]
                Result = SIR_EAKF(MSA_sam_record[index_t][j][0], MSA_sam_record[index_t][j][1],MSA_sam_record[index_t][j][2],n, 
                                    MSA_sam_record[index_t][j][0]+MSA_sam_record[index_t][j][1]+MSA_sam_record[index_t][j][2], 
                                    C, beta, L, D, gamma,fly,index_t+real_time,edge_matrix)
                MSA_sam_record[index_t + 1,j,:,:]=Result[0]
                MSA_sam_record_1[index_t + 1][j][0]=Result[2]
                all_numens_i_commute_fly[0,:,:]=all_numens_i_commute_fly[0,:,:]+Result[4]/num_ens
                all_numens_i_commute_fly[1,:,:]=all_numens_i_commute_fly[1,:,:]+Result[5]/num_ens
                N_All[index_t+1][j]=Result[1]
                sim_case[t] = MSA_sam_record_1[index_t+1,:,0,:]
                
        week_case = np.sum(sim_case, axis=0)
        para_record[week + 1] = para_record[week]
        for k in MSA_observe:
            if np.var(week_case[:, k]) == 0:
                post_var = 0.1
                prior_var = 0.1
            else:
                prior_var = np.var(week_case[:, k])
                post_var = prior_var * obs_var[k] / (prior_var + obs_var[k])
            prior_mean = np.mean(week_case[:, k])
            post_mean = post_var * (prior_mean / prior_var + truth[k] / obs_var[k])
            alpha = (obs_var[k] / (obs_var[k] + prior_var)) ** 0.5
            delta = post_mean + alpha * (week_case[:, k] - prior_mean) - week_case[:, k]
            post_case[week][k] = week_case[:, k] + delta
            for i in range(3):
                corr = (np.cov([MSA_sam_record[index_t + 1, :, i, k], week_case[:, k]])[0][1]) / prior_var
                MSA_sam_record[index_t + 1, :, i, k] += corr * delta
                MSA_sam_record[index_t + 1, :, i, k] = checkBound(MSA_sam_record[index_t + 1, :, i, k], i, k,num_ens,N_All[index_t+1])
            MSA_sam_record[index_t + 1, :, :, k] = Getstandard(MSA_sam_record[index_t + 1, :, :, k], num_ens, k,N_All[index_t+1])
            for i in range(1):
                corr = (np.cov([para_record[week + 1, :, k], week_case[:, k]])[0][1]) / prior_var
                para_record[week + 1, :, k] += corr * delta
                para_record[week + 1, :, k] = checkParaBound(para_record[week + 1, :, k], i, p0, num_ens)
    predic_state = np.array([MSA_sam_record[7 + 7 * k1] for k1 in range(wnum - 1)])
    predic_para = para_record
    return predic_para, predic_state, all_numens_i_commute_fly


#Initialization
def getStart_inference(num_ens,N_all,seed_loc):
    StateStart=np.zeros((num_ens,3,len(N_all)))
    for i in range(num_ens):
        for j in range(len(N_all)):
            StateStart[i][1][j]=0
            StateStart[i][2][j]=0
            StateStart[i][0][j]=N_all[j]-StateStart[i][1][j]-StateStart[i][2][j]
        for j in seed_loc:
            seed_num=random.randint(10,30)
            StateStart[i][1][j]=seed_num
            StateStart[i][2][j]=0
            StateStart[i][0][j]=N_all[j]-StateStart[i][1][j]-StateStart[i][2][j]
    return StateStart

def getParaStart(num_ens,p0,beta):
    p0=int(p0)
    paraStart=np.zeros((num_ens,p0))
    for i in range(num_ens):
        for j in range(p0):
            paraStart[i][j]=random.uniform(beta[0],beta[1])
    return  paraStart

#Onset
def max_loc(truth):
    location_list = []
    week = 5
    for i in range(len(truth)-week):
        if truth[i+1] < truth[i] and truth[i+2] < truth[i] and truth[i+3] < truth[i] and truth[i+4] < truth[i] and truth[i+5] < truth[i] and truth[i]>10:
            location_list.append(i)
    if len(location_list) == 0:
        part_truth = [truth[-1],truth[-2],truth[-3],truth[-4],truth[-5]]
        max_loc =  np.argmax(part_truth)
        location_list.append(int(len(truth)-1-max_loc))
    return location_list

def over_1000(data):
    for index, value in enumerate(data):
        if value >= 2000:
            return index
    return 100

def optimize_parameters(truth, first_zero, T_j, seed_set, week_num):
    
    def fun(para, x, y, t_j):
        n = len(x)
        residuals = y - (para[0] + para[1] * x + para[2] * np.maximum((x - t_j), 0))
        sigma = 1
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma) - 0.5 * np.sum(residuals**2) / sigma
        return -(log_likelihood)
    timelist = []
    betalist = []
    truth_max = []
    for i in range(truth.shape[1]):
        if i in seed_set:
            timelist.append(100)
            betalist.append([0,0,0])
            continue
        loc_week = truth[first_zero[i]:,i]
        if np.max(loc_week) > 0:
            break_1000 = over_1000(loc_week)
            if break_1000 != 100:
                loc = min(np.min(max_loc(loc_week)),break_1000)
            else:
                loc = np.min(max_loc(loc_week))
            truth_max.append(first_zero[i]+loc)
        else:
            timelist.append(100)
            betalist.append([0,0,0])
            truth_max.append(0)
            continue
        if loc < week_num:
            week = loc
        else:
            week = week_num
        L1 = []
        paralist = []
        for t_j in T_j:
            x = np.arange(first_zero[i]+loc - week, first_zero[i]+loc + 1)
            y = truth[:, i][first_zero[i]+loc - week:first_zero[i]+loc + 1]
            x0 = (0.2, 0.2, 0.2)
            bounds = [(None, None), (1e-5, None), (1e-5, None)]
            res = minimize(fun, x0, args=(x, y, t_j), method='Nelder-Mead', bounds=bounds)
            paralist.append(res.x)
            L1.append(-res.fun)
        max_point = np.argmax(L1)
        timelist.append(T_j[max_point])
        betalist.append(paralist[max_point])
        
    return timelist


job_id=get_slurm_job_id()

#Load infection data and population data
MSA_small_real_case=np.array(pd.read_csv('MSA_real_case_free_simulation.csv'))
N_all_small_scale=np.array(pd.read_csv('MSA_pop.csv')).reshape(-1)

#Load commuting data 
C_small_scale=np.array(pd.read_csv("Daily_commuting_MSA_free_simulation.csv"))

#Load Flight data 
fly=sparse.load_npz("fly_free_simulation.npz")
fly=fly.toarray().reshape(730,369,369)

#Set the parameters required for inference
obs_power=2
real_time=0
num_ens=100
gamma=2.36
eps=1e-8
p_value=0.05
beta_matrix = np.zeros(2)
beta_matrix[0]=0.3
beta_matrix[1]=0.45

#In the inference, each location had 3 extra attempts to infer infection source.
week_error=3

#Set infection source
seed_loc=[187]

#Calculate onset and set the inference sequence
onset_list= optimize_parameters(MSA_small_real_case, [0 for i in range(369)],list(np.arange(0,25,1/7)), [], 25)
time_list = pd.DataFrame(np.zeros((len(N_all_small_scale),2)))
time_list.iloc[:,0] = [int(i) for i in range(len(N_all_small_scale))]
time_list.iloc[:,1] = onset_list
time_list = np.array(time_list.sort_values(by=time_list.columns[1], ascending=True))
inference_list=list(time_list[:,0])
for i in range(len(inference_list)):
    inference_list[i]=int(inference_list[i])
seed_threshold=np.floor(onset_list).reshape(len(N_all_small_scale),1)
breakout_time_series=np.zeros(len(time_list))
for i in range(len(breakout_time_series)):
    row=time_list[i][0]
    breakout_time_series[int(row)]=i
max_onset=np.max(seed_threshold[seed_threshold<100])
min_onset=np.min(seed_threshold)

#Exclude all MSAs that require inference
all_possible=[]
for id in range(len(C_small_scale)):
    if np.sum(MSA_small_real_case[:,id])>0:
        all_possible.append(id)
        
#Load Groundtruth
Ground_truth=pd.read_csv('Ground_truth_free_simulation.csv').values

#inference
right=0
false=0

infection_source=seed_loc.copy()
infection_object=[k for k in all_possible if k not in infection_source]
current_time=int(min_onset)
infection_state=np.zeros([len(C_small_scale)*5,len(C_small_scale),len(C_small_scale)])
l=0
infection_state[l]=np.diag(np.ones(len(C_small_scale)))
for i in infection_source:
    for j in infection_source:
        infection_state[l][i][j]=1
infection_inference_path=np.zeros([len(N_all_small_scale),len(N_all_small_scale)])
infection_inference_path_transmission=np.zeros([len(N_all_small_scale),len(N_all_small_scale)])

#According to onset make inferences
for m in range(int(max_onset+week_error)):
    #Infection source exists and there is an MSA whose source has not been determined within the prescribed time
    if len(infection_source)>0 and len(infection_object)!=0 :
        #Find all MSAs whose source may been determined in current time
        infection_object_possible=[]
        judgement_flag=0
        for i in inference_list:
            if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-week_error and i in infection_object:
                add_command=0
                for j in infection_source:
                    if (C_small_scale[i][j]+C_small_scale[j][i])>0 or (np.sum(fly[real_time:real_time+(current_time+1)*7,i,j])+np.sum(fly[real_time:real_time+(current_time+1)*7,j,i]))>0:
                        add_command=1
                if add_command==1:
                    infection_object_possible.append(i)
                    add_command=0
        infection_object_possible_time=breakout_time_series[infection_object_possible]
        infection_object_possible_time=infection_object_possible_time.tolist()
        current_edge_state=infection_state[l]
        object_times=len(infection_object_possible)
        while len(infection_object_possible)>0:
            #Identify the MSA with an earlier onset
            time_break=min(infection_object_possible_time)
            time_num=infection_object_possible_time.index(time_break)
            j=time_num
            w_num=max(int(current_time),2)
            infection_source_possible=[]
            for i in infection_source:
                add=0
                if (C_small_scale[infection_object_possible[j]][i]+C_small_scale[i][infection_object_possible[j]])>0:
                    add=1
                if (np.sum(fly[real_time:real_time+(w_num+1)*7,i,infection_object_possible[j]])+np.sum(fly[real_time:real_time+(w_num+1)*7,infection_object_possible[j],i]))>0:
                    add=1
                if add==1:
                    infection_source_possible.append(i)
            
            #Set initial state, transmission parameter beta, infectious period and immunity period     
            sam_0=getStart_inference(num_ens,N_all_small_scale,seed_loc)
            p0=len(N_all_small_scale)
            para_0=getParaStart(num_ens, p0,beta_matrix)
            D_sam=[]
            L_sam=[]            
            for i in range(num_ens):
                D_sam.append(np.random.uniform(3,5))
                L_sam.append(np.random.uniform(365,2*365))
                
            stop_command=0
            add_edge_flag=0
            error=MSA_small_real_case[w_num,infection_object_possible[j]]*np.ones(num_ens)
            error_possible=np.zeros([num_ens,len(infection_source_possible)])
            predict=[]
            likelihood=np.zeros([len(C_small_scale),len(C_small_scale)])
            likelihood=pd.DataFrame(likelihood)
            ratio_value=np.ones([len(C_small_scale),len(C_small_scale)])*(-1000)
            ratio_value=pd.DataFrame(ratio_value)
            Transmission=np.zeros([len(C_small_scale),len(C_small_scale)])
            Transmission=pd.DataFrame(Transmission)
            
            #Parallel inference
            def inference(current_edge_state,i,w_num,error):
                start_time = time.time()
                np.random.seed(int(start_time+i))
                current_edge_state_0=current_edge_state.copy()
                current_edge_state_0[infection_object_possible[j]][infection_source_possible[i]]=1
                current_edge_state_0[infection_source_possible[i]][infection_object_possible[j]]=1
                C_small_scale_assumption=C_small_scale*current_edge_state_0
                small_scale_seed=infection_source.copy()
                small_scale_seed.append(infection_object_possible[j])
                C_small_scale_assumption_modification=C_small_scale_assumption[small_scale_seed,:]
                C_small_scale_assumption_modification=C_small_scale_assumption_modification[:,small_scale_seed]
                fly_modification=fly[:,small_scale_seed,:]
                fly_modification=fly_modification[:,:,small_scale_seed]
                current_edge_state_0_modification=current_edge_state_0[small_scale_seed,:]
                current_edge_state_0_modification=current_edge_state_0_modification[:,small_scale_seed]
                #EAKF fitting
                predict_para, predict_state,all_numens_i_commute_fly= train_EAKF(MSA_small_real_case[:,small_scale_seed], (w_num+1) * 7,
                                                                            obs_power, sam_0[:,:,small_scale_seed], para_0[:,small_scale_seed], num_ens, w_num+1 , p0, 
                                                                                N_all_small_scale[small_scale_seed], 
                                                                            C_small_scale_assumption_modification,fly_modification,current_edge_state_0_modification,
                                                                            real_time,D_sam,L_sam)
                #Simulation predicting
                beta_estimate=predict_para[-1:,:,:][0]
                state_estimate=predict_state[-1:,:,:,:][0]
                predict_sample=np.zeros(num_ens)
                for u in range(num_ens):
                    N_current_predict=state_estimate[u,0,:]+state_estimate[u,1,:]+state_estimate[u,2,:]
                    free_predict=SIR_simulation_predict(state_estimate[u,0,:],state_estimate[u,1,:],state_estimate[u,2,:],len(N_all_small_scale[small_scale_seed]),
                                                        7+1,N_current_predict,C_small_scale_assumption_modification,beta_estimate[u,:],L_sam[u],D_sam[u],gamma,
                                                        fly_modification,real_time+(w_num)*7,current_edge_state_0_modification)
                    I_increase_estiamte=free_predict[5]
                    all_numens_i_commute_fly=all_numens_i_commute_fly+free_predict[8]/num_ens
                    predict_sample[u]=(np.sum(I_increase_estiamte[-7:,:],axis=0))[small_scale_seed.index(infection_object_possible[j])]
                current_edge_state_0[int(infection_object_possible[j])][infection_source_possible[i]]=0
                current_edge_state_0[infection_source_possible[i]][int(infection_object_possible[j])]=0
                #get significance, mae and transmission way
                real_MSA=MSA_small_real_case[w_num,infection_object_possible[j]]
                predict_MSA=predict_sample
                for row in range(len(all_numens_i_commute_fly[0,:,:])):
                    all_numens_i_commute_fly[0,row,row]=0
                    all_numens_i_commute_fly[1,row,row]=0
                i_commute_number=np.sum(all_numens_i_commute_fly[0,:,:],axis=0)[small_scale_seed.index(infection_object_possible[j])]
                i_fly_number=np.sum(all_numens_i_commute_fly[1,:,:],axis=0)[small_scale_seed.index(infection_object_possible[j])]
                if i_commute_number>i_fly_number:
                    transmission=1
                if i_commute_number==i_fly_number:
                    transmission=2
                if i_commute_number<i_fly_number:
                    transmission=3
                if np.sum(predict_MSA) != 0:
                    error_sample=predict_MSA-real_MSA
                    t,p =st.mannwhitneyu(np.abs(error_sample),np.abs(error),alternative='less')
                    mae=np.abs(np.median(error))-np.abs(np.median(error_sample))
                    predict_sample=np.mean(predict_MSA)
                else:
                    error_sample=np.abs(error)
                    predict_sample=0
                    t=0.1
                    p=1
                    mae=0
                return mae, p , i, error_sample,predict_sample,transmission
            
            if __name__ == '__main__':
                with concurrent.futures.ProcessPoolExecutor(max_workers=35) as executor:
                    futures={executor.submit(inference,current_edge_state,i,w_num,error): i for i in range(len(infection_source_possible))}
                    for future in concurrent.futures.as_completed(futures):
                        mae, p , i, error_sample, predict_sample, transmission = future.result()
                        likelihood[infection_object_possible[j]][infection_source_possible[i]]=p
                        ratio_value[infection_object_possible[j]][infection_source_possible[i]]=mae
                        error_possible[:,i]=error_sample
                        Transmission[infection_object_possible[j]][infection_source_possible[i]]=transmission
            likelihood=np.array(likelihood)
            ratio_value=np.array(ratio_value)
            Transmission=np.array(Transmission)
            ratio_choice=np.where(ratio_value==np.max(ratio_value[:,infection_object_possible[j]]))[0][0]
            
            #The decision-making of source
            if np.max(ratio_value[:,infection_object_possible[j]])>0 and likelihood[int(ratio_choice),infection_object_possible[j]]<p_value:
                path_choice_1=np.where(ratio_value==np.max(ratio_value[:,infection_object_possible[j]]))[0][0]
                path_choice_2=np.where(ratio_value==np.max(ratio_value[:,infection_object_possible[j]]))[1][0]
                current_edge_state[path_choice_1][path_choice_2]=1
                current_edge_state[path_choice_2][path_choice_1]=1
                infection_state[l+1]=current_edge_state.copy()
                infection_inference_path[path_choice_1][path_choice_2]=1
                l=l+1
                if Ground_truth[int(path_choice_1),int(infection_object_possible[j])] == 1:
                    right=right+1
                else:
                    false=false+1
                infection_inference_path_transmission[path_choice_1][path_choice_2]=Transmission[int(path_choice_1),int(infection_object_possible[j])]
                infection_source_possible.remove(path_choice_1)
            else:
                #The source is not found
                add_edge_flag=1
    
            if add_edge_flag==0:
                #Attemptations when reliable infection source was found
                infection_source.append(infection_object_possible[j])
                infection_object.remove(infection_object_possible[j])
                judgement_flag=0
                for i in inference_list:
                    if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-week_error and i in infection_object and i not in infection_object_possible:
                        add_command=0
                        for j0 in infection_source:
                            if (C_small_scale[i][j0]+C_small_scale[j0][i])>0 or (np.sum(fly[real_time:real_time+(w_num+1)*7,i,j0])+np.sum(fly[real_time:real_time+(w_num+1)*7,j0,i]))>0:
                                add_command=1
                        if add_command==1:
                            infection_object_possible.append(i)
                            add_command=0
                infection_object_possible.remove(infection_object_possible[j])
                infection_object_possible_time=breakout_time_series[infection_object_possible]
                infection_object_possible_time=infection_object_possible_time.tolist()
            else:
                #Attemptations when no reliable infection source was found
                judgement_flag=0
                if seed_threshold[int(infection_object_possible[j])][0]<=current_time-week_error :
                    judgement_flag=1
                    infection_source.append(int(infection_object_possible[j]))
                    if seed_threshold[int(infection_object_possible[j])][0]<=4:
                        judgement_flag=2
                        seed_loc.append(int(infection_object_possible[j]))
                infection_object.remove(infection_object_possible[j])
                if judgement_flag==2:
                    for i in inference_list:
                        if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-week_error and i in infection_object and i not in infection_object_possible:
                            add_command=0
                            for j0 in infection_source:
                                if (C_small_scale[i][j0]+C_small_scale[j0][i])>0 or (np.sum(fly[real_time:real_time+(w_num+1)*7,i,j0])+np.sum(fly[real_time:real_time+(w_num+1)*7,j0,i]))>0:
                                    add_command=1
                            if add_command==1:
                                infection_object_possible.append(i)
                                add_command=0
                infection_object_possible.remove(infection_object_possible[j])
                infection_object_possible_time=breakout_time_series[infection_object_possible]
                infection_object_possible_time=infection_object_possible_time.tolist()
                                            
        if len(infection_object_possible)<1:
            current_time=current_time+1
    infection_object=[k for k in all_possible if k not in infection_source]



infection_inference_path=pd.DataFrame(infection_inference_path)
infection_inference_path.to_excel('free_simulation_path'+str(job_id)+'.xlsx',index=None)
infection_inference_path_transmission=pd.DataFrame(infection_inference_path_transmission)
infection_inference_path_transmission.to_excel('free_simulation_transmission'+str(job_id)+'.xlsx',index=None)





