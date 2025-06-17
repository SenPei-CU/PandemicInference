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

#SEIR Model
def SEIR_simulation_predict(s,e,i,r,n,t,N,C,beta,L,D,gamma,fly,fly_time,edge_matrix,commuting_time,Z):
    flag=0
    state=np.zeros([4,t,n])
    new_increase_I=np.zeros([t,n])
    MSA_pop=np.zeros([t,n])
    state[0,0,:]=s
    state[1,0,:]=e
    state[2,0,:]=i
    state[3,0,:]=r
    MSA_pop[0][:]=N
    i_commute_fly=np.zeros([2,n,n])
    for k in range(t-1):
        C_rate=C[commuting_time]/np.sum(C[commuting_time],axis=1).reshape(n,1)
        M=np.zeros([len(N),len(N)+1])
        M[:,0:len(N)]=(fly[int(fly_time),:,:]*edge_matrix)/np.sum(C[commuting_time],axis=1).reshape(len(N),1)
        commuting_time=commuting_time+1
        fly_time=fly_time+1
        state_Flow_Matrix_multinomial=np.zeros([4,len(N),len(N)])
        for u in range(n):
           state_Flow_Matrix_multinomial[0,u,0:len(N)]=np.random.multinomial(state[0][k][u],M[u,:])[0:len(N)]
           state_Flow_Matrix_multinomial[1,u,0:len(N)]=np.random.multinomial(state[1][k][u],M[u,:])[0:len(N)]
           state_Flow_Matrix_multinomial[2,u,0:len(N)]=np.random.multinomial(state[2][k][u],M[u,:])[0:len(N)]
           state_Flow_Matrix_multinomial[3,u,0:len(N)]=np.random.multinomial(state[3][k][u],M[u,:])[0:len(N)]
        sum_s_in = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=0)
        sum_s_out = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=1)
        sum_e_in = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=0)
        sum_e_out = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=1)
        sum_i_in = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=0)
        sum_i_out = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=1)
        sum_r_in = np.sum(state_Flow_Matrix_multinomial[3,:,:],axis=0)
        sum_r_out = np.sum(state_Flow_Matrix_multinomial[3,:,:],axis=1)
        sum_new_increase_matrix = np.zeros([n,n])
        S_rate=state[0][k].reshape(n,1)*C_rate
        delta_I=np.sum(beta*(state[2][k]/MSA_pop[k,:])*S_rate,axis=0)
        delta_I_nb=np.zeros(n)
        p=(1/np.sum(S_rate,axis=0))*S_rate
        nonzero_indices = np.nonzero(delta_I)
        choice = list(nonzero_indices[0])
        test_times=gamma*state[2,k,:]
        test_times[test_times==0]=0.000000001
        delta_I_nb=np.random.negative_binomial(test_times,test_times/(test_times+delta_I))
        for u in choice:
            sum_new_increase_matrix[u,:]=np.random.multinomial(int(delta_I_nb[u]),p[:,u])
        sum_new_increase=np.sum(sum_new_increase_matrix,axis=0)
        e_to_i_Vector=np.random.poisson(state[1,k,:] / Z,size=n)
        r_to_s_Vector=np.random.poisson(state[3,k,:] / L,size=n)
        i_to_r_Vector=np.random.poisson(state[2,k,:] / D,size=n)
        state[0,k+1,:] = state[0,k,:] - sum_new_increase + r_to_s_Vector + (sum_s_in - sum_s_out)
        state[1,k+1,:] = state[1,k,:] + sum_new_increase - e_to_i_Vector + (sum_e_in - sum_e_out)
        state[2,k+1,:] = state[2,k,:] + e_to_i_Vector -i_to_r_Vector + (sum_i_in - sum_i_out)
        state[3,k+1,:] = state[3,k,:] + i_to_r_Vector - r_to_s_Vector +(sum_r_in - sum_r_out)
        N = state[0,k+1,:]+state[1,k+1,:]+state[2,k+1,:]+state[3,k+1,:]
        MSA_pop[k + 1][:] = N
        new_increase_I[k+1,:]=sum_new_increase
        i_commute_fly[0,:,:]=i_commute_fly[0,:,:]+sum_new_increase_matrix
        i_commute_fly[1,:,:]=i_commute_fly[1,:,:]+state_Flow_Matrix_multinomial[1,:,:]+state_Flow_Matrix_multinomial[2,:,:]
        if np.min(state[:,k+1,:])<0:
            flag=1
            for i in range(len(N)):
                if np.min(state[:,k+1,i])<0:
                    sum=state[0,k+1,i]+state[1,k+1,i]+state[2,k+1,i]
                    for j in range(4):
                        if state[j][k+1][i]<0:
                            state[j][k+1][i]=1
                    weight=state[:,k+1,i]/sum
                    state[:,k+1,i]=sum*weight
                    for j in range(4):
                        state[j][k+1][i]=int(state[j][k+1][i])
  
    return state[0],state[1],state[2],state[3],MSA_pop,N,new_increase_I,flag,fly_time,i_commute_fly

def SEIR_EAKF(s,e,i,r,n,N,C,beta,L,D,gamma,fly,fly_time,edge_matrix,Z):
    flag=0
    state=np.zeros([4,n])
    new_increase_I=np.zeros(n)
    C_rate=C/np.sum(C,axis=1).reshape(n,1)
    M=np.zeros([n,n+1])
    for k in range(1):
        M[:,0:n]=(fly[int(fly_time),:,:]*edge_matrix)/np.sum(C,axis=1).reshape(n,1)
        state_Flow_Matrix_multinomial=np.zeros([4,n,n])
        for u in range(n):
           state_Flow_Matrix_multinomial[0,u,0:n]=np.random.multinomial(s[u],M[u,:])[0:n]
           state_Flow_Matrix_multinomial[1,u,0:n]=np.random.multinomial(e[u],M[u,:])[0:n]
           state_Flow_Matrix_multinomial[2,u,0:n]=np.random.multinomial(i[u],M[u,:])[0:n]
           state_Flow_Matrix_multinomial[3,u,0:n]=np.random.multinomial(r[u],M[u,:])[0:n]
        sum_s_in = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=0)
        sum_s_out = np.sum(state_Flow_Matrix_multinomial[0,:,:],axis=1)
        sum_e_in = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=0)
        sum_e_out = np.sum(state_Flow_Matrix_multinomial[1,:,:],axis=1)
        sum_i_in = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=0)
        sum_i_out = np.sum(state_Flow_Matrix_multinomial[2,:,:],axis=1)
        sum_r_in = np.sum(state_Flow_Matrix_multinomial[3,:,:],axis=0)
        sum_r_out = np.sum(state_Flow_Matrix_multinomial[3,:,:],axis=1)
        sum_new_increase_matrix = np.zeros([n,n])
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
            sum_new_increase_matrix[u,:]=np.random.multinomial(int(delta_I_nb[u]),p[:,u])
        sum_new_increase=np.sum(sum_new_increase_matrix,axis=0)
        e_to_i_Vector=np.random.poisson(e / Z,size=n)
        r_to_s_Vector=np.random.poisson(r / L,size=n)
        i_to_r_Vector=np.random.poisson(i / D,size=n)
        state[0,:] = s - sum_new_increase + r_to_s_Vector + (sum_s_in - sum_s_out)
        state[1,:] = e + sum_new_increase - e_to_i_Vector + (sum_e_in - sum_e_out)
        state[2,:] = i + e_to_i_Vector -i_to_r_Vector + (sum_i_in - sum_i_out)
        state[3,:] = r + i_to_r_Vector - r_to_s_Vector +(sum_r_in - sum_r_out)
        N = state[0,:]+state[1,:]+state[2,:]+state[3,:]
        new_increase_I=sum_new_increase
        if np.min(state)<0:
            flag=1
            for i in range(len(N)):
                if np.min(state[:,i])<0:
                    sum=state[0,i]+state[1,i]+state[2,i]+state[3,i]
                    for j in range(4):
                        if state[j][i]<0:
                            state[j][i]=1
                    weight=state[:,i]/sum
                    state[:,i]=sum*weight
                    for j in range(4):
                        state[j][i]=int(state[j][i])
        fly_transmission=state_Flow_Matrix_multinomial[1,:,:]+state_Flow_Matrix_multinomial[2,:,:]
    return state,N,new_increase_I,flag,sum_new_increase_matrix,fly_transmission

#EAKF
def Getstandard(prior_sam_state,num_ens,uhf_id,N_all):
    post_sam_state=np.zeros((num_ens,4))
    for i in range(num_ens):
        for j in range(4):
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
        if post_data[prior_data > 1.7].size > 0 or post_data[prior_data < 0].size > 0:
            for i in range(num_ens):
                if prior_data[i] > 1.7:
                    post_data[i] = random.uniform(1.6, 1.7)
                if prior_data[i] < 1.3:
                    post_data[i] =  random.uniform(1.3, 1.4)

    else:
        if post_data[prior_data > 2 * p0[flag]].size > 0 or post_data[prior_data < 2].size > 0:
            for i in range(num_ens):
                if prior_data[i] < 2:
                    post_data[i] =  np.mean(prior_data)
                if prior_data[i] > 2 * p0[flag]:
                    post_data[i] = min(random.uniform(1.5, 1.6) * np.mean(prior_data),
                                       random.uniform(1.5, 1.6) * p0[flag])
    return post_data

def train_EAKF(MSA_real_case,T,obs_power,sam_0,para_0,num_ens,p0,N_all,C,fly,edge_matrix,real_time,D_sam,L_sam,Z_sam):
    MSA_observe=[]
    all_numens_i_commute_fly=np.zeros([2,len(N_all),len(N_all)])
    for id in range(len(N_all)):
        if np.sum(MSA_real_case[:,id])>0:
            MSA_observe.append(id)
    MSA_sam_record=np.zeros((T+1,num_ens,4,len(N_all)))
    MSA_sam_record_1=np.zeros((T+1,num_ens,1,len(N_all)))
    para_record=np.zeros((T+1,num_ens,len(N_all)))
    MSA_sam_record[0]=sam_0
    para_record[0]=para_0
    post_case=np.zeros((T,len(N_all),num_ens))
    N_All=np.zeros([T+1,num_ens,len(N_all)])
    for i in range(num_ens):
        N_All[0][i]=N_all
    sim_case = np.zeros((T,num_ens,len(N_all)))
    for index_t in range(T):
        truth = MSA_real_case[index_t]
        oevbase = (10) ** 2  
        obs_var = np.array(oevbase + truth ** obs_power / 100 )  
        # EAKF
        for j in range(num_ens):  
            n=len(N_all)
            beta_id=[k for k in range(n)]
            beta=para_record[index_t,j,beta_id]
            D=D_sam[j]
            gamma=0.55
            L=L_sam[j]
            Z=Z_sam[j]
            Result = SEIR_EAKF(MSA_sam_record[index_t][j][0], MSA_sam_record[index_t][j][1],MSA_sam_record[index_t][j][2],MSA_sam_record[index_t][j][3],n, 
                                MSA_sam_record[index_t][j][0]+MSA_sam_record[index_t][j][1]+MSA_sam_record[index_t][j][2]+MSA_sam_record[index_t][j][3], 
                                C[index_t,:,:], beta, L, D, gamma,fly,index_t+real_time,edge_matrix,Z)
            MSA_sam_record[index_t + 1,j,:,:]=Result[0]
            MSA_sam_record_1[index_t + 1][j][0]=Result[2]
            all_numens_i_commute_fly[0,:,:]=all_numens_i_commute_fly[0,:,:]+Result[4]/num_ens
            all_numens_i_commute_fly[1,:,:]=all_numens_i_commute_fly[1,:,:]+Result[5]/num_ens
            N_All[index_t+1][j]=Result[1]
        sim_case[index_t] = MSA_sam_record_1[index_t+1,:,0,:]
        week_case = sim_case[index_t]
        para_record[index_t + 1] = para_record[index_t]
        
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
            post_case[index_t][k] = week_case[:, k] + delta
            for i in range(4):
                corr = (np.cov([MSA_sam_record[index_t + 1, :, i, k], week_case[:, k]])[0][1]) / prior_var
                MSA_sam_record[index_t + 1, :, i, k] += corr * delta
                MSA_sam_record[index_t + 1, :, i, k] = checkBound(MSA_sam_record[index_t + 1, :, i, k], i, k,num_ens,N_All[index_t+1])
            MSA_sam_record[index_t + 1, :, :, k] = Getstandard(MSA_sam_record[index_t + 1, :, :, k], num_ens, k,N_All[index_t+1])
            for i in range(1):
                corr = (np.cov([para_record[index_t + 1, :, k], week_case[:, k]])[0][1]) / prior_var
                para_record[index_t + 1, :, k] += corr * delta
                para_record[index_t + 1, :, k] = checkParaBound(para_record[index_t + 1, :, k], i, p0, num_ens)
    predic_state = MSA_sam_record
    predic_para = para_record
    return predic_para, predic_state, all_numens_i_commute_fly

#Initialization
def getStart_inference(num_ens,N_all,seed_loc,retio):
    StateStart=np.zeros((num_ens,4,len(N_all)))
    for i in range(num_ens):
        for j in range(len(N_all)):
            StateStart[i][1][j]=0
            StateStart[i][2][j]=0
            StateStart[i][3][j]=0
            StateStart[i][0][j]=N_all[j]-StateStart[i][1][j]-StateStart[i][2][j]-StateStart[i][3][j]

        seed_all=np.random.randint(500,1000)
        for j in seed_loc:
            StateStart[i][2][j]=np.round(retio[j]*seed_all)
            StateStart[i][1][j]=StateStart[i][1][j]*5
            StateStart[i][3][j]=0
            StateStart[i][0][j]=N_all[j]-StateStart[i][1][j]-StateStart[i][2][j]-StateStart[i][3][j]
    return StateStart

def getParaStart(num_ens,p0,beta):
    p0=int(p0)
    paraStart=np.zeros((num_ens,p0))
    for i in range(num_ens):
        for j in range(int(p0)):
            paraStart[i][j]=random.uniform(beta[0],beta[1])
    return paraStart

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
        if value >= 1000:
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

#Load population data and infection data
MSA_list=pd.read_excel('MSA_list_SARS-CoV-2.xlsx')
MSA_small_real_case=np.array(pd.read_csv('DailyInfection_MSA_SARS-CoV-2.csv'))


#Load commuting data 
C_small_scale=sparse.load_npz("Daily_commuting_MSA_SARS-CoV-2.npz")
C_small_scale=C_small_scale.toarray().reshape(345,377,377)

#Load Flight data 
fly=sparse.load_npz("fly_SARS-CoV-2.npz")
fly=fly.toarray().reshape(730,377,377)

#Set the parameters required for inference
obs_power=2
real_time=51
num_ens=100
gamma=0.55
eps=1e-8
p_value=0.05
beta_matrix = np.zeros(2)
beta_matrix[0]=1.3
beta_matrix[1]=1.7

#In the inference, each location had 7 extra attempts to infer infection source.
day_error=7

#Set transmission parameter beta, infectious period, immunity period and latency period
    
#Set infection sources and their initial state
seed_num=[219,353]
y=[]
sum_MSA_real_case=np.cumsum(MSA_small_real_case,axis=0)
for i in range(377):
    if i in seed_num:
        y.append(sum_MSA_real_case[13,i])
    else:
        y.append(0)
retio=y/np.sum(y)
N_all_small_scale=np.array(MSA_list.loc[:,'MSA people'])

#Load Onset data and set the inference sequence
day_time = 80
x=[0 for i in range(day_time)]
for i in range(377):
    if np.max(MSA_small_real_case[:,i]) <10:
        MSA_small_real_case[:,i] = x
seed_set = []
first_non_zero_indices =[0 for i in range(377)]
T_j = list(np.arange(0,day_time,1))
onset_list = optimize_parameters(MSA_small_real_case, first_non_zero_indices, T_j, seed_set, day_time-1)
timelist = pd.DataFrame(np.zeros((377,2)))
index_list = [int(i) for i in range(377)]
timelist.iloc[:,0] = index_list
timelist.iloc[:,1] = onset_list
timelist = timelist.sort_values(by=timelist.columns[1], ascending=True)
timelist=np.array(timelist)
for i in range(len(timelist)):
    timelist[i][0]=int(timelist[i][0])
inference_list=list(timelist[:,0])
for i in range(len(inference_list)):
    inference_list[i]=int(inference_list[i])
seed_threshold=np.floor(onset_list).reshape(len(MSA_list),1)
breakout_time_series=np.zeros(len(timelist))
for i in range(len(breakout_time_series)):
    row=timelist[i][0]
    breakout_time_series[int(row)]=i
max_onset=np.max(seed_threshold[seed_threshold<100])
min_onset=np.min(seed_threshold)

#Exclude all MSAs that require inference
all_possible=[]
for id in range(len(N_all_small_scale)):
    if np.sum(MSA_small_real_case[:,id])>0 and seed_threshold[id,0]<=80:
        all_possible.append(id)


#inference
infection_source=seed_num.copy()
infection_object=[k for k in all_possible if k not in infection_source]
current_time=int(min_onset)
infection_state=np.zeros([len(MSA_list)*5,len(MSA_list),len(MSA_list)])
l=0
infection_state[l]=np.diag(np.ones(len(MSA_list)))
for i in infection_source:
    for j in infection_source:
        infection_state[l][i][j]=1
infection_inference_path=np.zeros([len(N_all_small_scale),len(N_all_small_scale)])
infection_inference_path_transmission=np.zeros([len(N_all_small_scale),len(N_all_small_scale)])
sim_t=7
#According to onset make inferences
for m in range(int(max_onset+day_error)):
    #Infection source exists, there is an MSA whose source has not been determined within the prescribed time
    #we only consider the transmission in the fist 80 days  
    if len(infection_source)>0 and len(infection_object)!=0 and current_time+sim_t<80:
        #Find all MSAs whose source may been determinede in current time
        infection_object_possible=[]
        judgement_flag=0
        for i in inference_list:
            if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-day_error and i in infection_object:
                add_command=0
                for j in infection_source:
                    if (np.sum(C_small_scale[0:(current_time+sim_t),i,j])+np.sum(C_small_scale[0:(current_time+sim_t),j,i]))>0 or (np.sum(fly[real_time:real_time+(current_time+sim_t),i,j])+np.sum(fly[real_time:real_time+(current_time+sim_t),j,i]))>0:
                        add_command=1
                if add_command==1:
                    infection_object_possible.append(i)
                    add_command=0
                    
        infection_object_possible_time=breakout_time_series[infection_object_possible]
        infection_object_possible_time=infection_object_possible_time.tolist()
        current_edge_state=infection_state[l]
        object_times=len(infection_object_possible)
        while len(infection_object_possible)>0:
            #Identify the MSA with a earlier onset 
            time_break=min(infection_object_possible_time)
            time_num=infection_object_possible_time.index(time_break)
            j=time_num
            w_num=current_time
            infection_source_possible=[]
            for i in infection_source:
                add=0
                if (np.sum(C_small_scale[0:w_num+sim_t,infection_object_possible[j],i])+np.sum(C_small_scale[0:w_num+sim_t,i,infection_object_possible[j]]))>0:
                    add=1
                if (np.sum(fly[real_time:real_time+w_num+sim_t,i,infection_object_possible[j]])+np.sum(fly[real_time:real_time+w_num+sim_t,infection_object_possible[j],i]))>0:
                    add=1
                if add==1:
                    infection_source_possible.append(i)
            
            #Set initial state, transmission parameter beta, infectious period, immunity period and latency period  
            sam_0=getStart_inference(num_ens,N_all_small_scale,seed_num,retio)
            p0=len(N_all_small_scale)
            para_0=getParaStart(num_ens, p0,beta_matrix)
            D_sam=[]
            L_sam=[]
            Z_sam=[]       
            for i in range(num_ens):
                D_sam.append(np.random.uniform(3,5))
                L_sam.append(np.random.randint(2*365,10*365))
                Z_sam.append(np.random.uniform(3,4))
            
            add_edge_flag=0
            error=MSA_small_real_case[w_num+sim_t,infection_object_possible[j]]*np.ones(num_ens)
            error_possible=np.zeros([num_ens,len(infection_source_possible)])
            predict=[]
            likelihood=np.zeros([len(MSA_list),len(MSA_list)])
            likelihood=pd.DataFrame(likelihood)
            ratio_value=np.ones([len(MSA_list),len(MSA_list)])*(-1000)
            ratio_value=pd.DataFrame(ratio_value)
            Transmission=np.zeros([len(MSA_list),len(MSA_list)])
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
                C_small_scale_assumption_modification=C_small_scale_assumption[:,small_scale_seed,:]
                C_small_scale_assumption_modification=C_small_scale_assumption_modification[:,:,small_scale_seed]
                fly_modification=fly[:,small_scale_seed,:]
                fly_modification=fly_modification[:,:,small_scale_seed]
                current_edge_state_0_modification=current_edge_state_0[small_scale_seed,:]
                current_edge_state_0_modification=current_edge_state_0_modification[:,small_scale_seed]
                #EAKF fitting
                predict_para, predict_state, all_numens_i_commute_fly= train_EAKF(MSA_small_real_case[:,small_scale_seed],  w_num+1,
                                                                            obs_power, sam_0[:,:,small_scale_seed], para_0[:,small_scale_seed], num_ens , p0 ,
                                                                            N_all_small_scale[small_scale_seed], 
                                                                            C_small_scale_assumption_modification,fly_modification,current_edge_state_0_modification,
                                                                            real_time,D_sam,L_sam,Z_sam)
                
                #Simulation predicting
                beta_estimate=predict_para[-1:,:,:][0]
                state_estimate=predict_state[-1:,:,:,:][0]
                predict_sample=np.zeros(num_ens)
                for u in range(num_ens):
                    N_current_predict=state_estimate[u,0,:]+state_estimate[u,1,:]+state_estimate[u,2,:]+state_estimate[u,3,:]
                    free_predict=SEIR_simulation_predict(state_estimate[u,0,:],state_estimate[u,1,:],state_estimate[u,2,:],state_estimate[u,3,:],len(N_all_small_scale[small_scale_seed]),
                                                            sim_t+1,N_current_predict,C_small_scale_assumption_modification,beta_estimate[u,:],L_sam[u],D_sam[u],gamma,
                                                            fly_modification,real_time+w_num+1,current_edge_state_0_modification,w_num+1,Z_sam[u])
                    I_increase_estiamte=free_predict[6]
                    all_numens_i_commute_fly=all_numens_i_commute_fly+free_predict[9]/num_ens
                    predict_sample[u]=I_increase_estiamte[sim_t][small_scale_seed.index(infection_object_possible[j])]
                current_edge_state_0[int(infection_object_possible[j])][infection_source_possible[i]]=0
                current_edge_state_0[infection_source_possible[i]][int(infection_object_possible[j])]=0
                #get significance, mae and transmission way
                real_MSA=MSA_small_real_case[w_num+sim_t,infection_object_possible[j]]
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
                if  np.sum(predict_MSA) != 0:
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
                        likelihood.loc[infection_source_possible[i],infection_object_possible[j]]=p
                        ratio_value.loc[infection_source_possible[i],infection_object_possible[j]]=mae
                        error_possible[:,i]=error_sample
                        Transmission.loc[infection_source_possible[i],infection_object_possible[j]]=transmission
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
                max_num=infection_source_possible.index(int(path_choice_1))
                error=np.abs(error_possible[:,max_num])
                infection_inference_path_transmission[path_choice_1][path_choice_2]=Transmission[int(path_choice_1),int(infection_object_possible[j])]
                infection_source_possible.remove(path_choice_1)
            else:
                #The Source is not found
                add_edge_flag=1

            if add_edge_flag==0:
                #Attemptations when reliable infection source was found
                infection_source.append(infection_object_possible[j])
                infection_object.remove(infection_object_possible[j])
                judgement_flag=0
                for i in inference_list:
                    if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-day_error and i in infection_object and i not in infection_object_possible:
                        add_command=0
                        for j0 in infection_source:
                            if (np.sum(C_small_scale[0:w_num+sim_t,i,j0])+np.sum(C_small_scale[0:w_num+sim_t,j0,i]))>0 or (np.sum(fly[real_time:real_time+w_num+sim_t,i,j0])+np.sum(fly[real_time:real_time+w_num+sim_t,j0,i]))>0:
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
                if seed_threshold[int(infection_object_possible[j])][0]<=current_time-day_error :
                    judgement_flag=1
                    infection_source.append(int(infection_object_possible[j]))
                    if seed_threshold[int(infection_object_possible[j])][0]<=12:
                        judgement_flag=2
                        y=[]
                        seed_num.append(int(infection_object_possible[j]))
                        for ii in range(377):
                            if ii in seed_num:
                                y.append(sum_MSA_real_case[13,ii])
                            else:
                                y.append(0)
                        retio=y/np.sum(y)
                infection_object.remove(infection_object_possible[j])
                if judgement_flag==2:
                    for i in inference_list:
                        if seed_threshold[i][0]<=current_time and seed_threshold[i][0]>=current_time-day_error and i in infection_object and i not in infection_object_possible:
                            add_command=0
                            for j0 in infection_source:
                                if (np.sum(C_small_scale[0:w_num+sim_t,i,j0])+np.sum(C_small_scale[0:w_num+sim_t,j0,i]))>0 or (np.sum(fly[real_time:real_time+w_num+sim_t,i,j0])+np.sum(fly[real_time:real_time+w_num+sim_t,j0,i]))>0:
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
infection_inference_path.to_excel('SARS-CoV-2_path'+str(job_id)+'.xlsx',index=None)
infection_inference_path_transmission=pd.DataFrame(infection_inference_path_transmission)
infection_inference_path_transmission.to_excel('SARS-CoV-2_transmission'+str(job_id)+'.xlsx',index=None)

