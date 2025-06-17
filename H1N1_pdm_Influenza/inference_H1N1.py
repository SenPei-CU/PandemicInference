import numpy as np 
import pandas as pd
import csv
import random 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import scipy.stats as st
import seaborn as sns
from statistics import median
from scipy.optimize import minimize
import concurrent.futures
import warnings
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import os

scale = 369
sam_num=100

def get_slurm_job_id(): 
  job_id = os.environ.get('SLURM_JOB_ID') 
  return job_id
result_id = get_slurm_job_id()
def SIR(y, N, beta, L, D, commuting_matrix, commuting_matrix_airport, scale):
    """SIR model implementation"""
    sum_row = np.sum(commuting_matrix, axis=1).reshape(scale, 1)
    M_ji = commuting_matrix / sum_row
    S_D = np.sum(M_ji * (y[0].reshape(scale, 1)), axis=0).reshape(1, scale)

    r = 2.36
    Ir = np.zeros((scale, 1))
    Im = np.zeros((scale, 1))
    delta_Ij = np.zeros((scale, 1))
    ps = np.zeros((scale, scale))
    ps = M_ji * (y[0].reshape(scale, 1)) / S_D
    seed_ij_1 = np.zeros((scale, scale))
    seed_ij_2 = np.zeros((3, scale, scale))
    num_SIR = np.zeros((3, scale))
    commuting_ratio = np.zeros((scale, scale + 1))
    commuting_ratio[:, 0:scale] = commuting_matrix_airport / sum_row
    
    no_nan_list = []
    for j in range(scale):
        if y[1][j] != 0:
            no_nan_list.append(j)
            
    for j in no_nan_list:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        Ir[j, 0] = y[1][j] * r 
        Im[j, 0] = y[1][j] * beta[j] * S_D[0, j] / N[j]
        prob = Ir[j, 0] / (Im[j, 0] + Ir[j, 0])
        pvals = np.array(ps[:, j])
        delta_Ij[j] = np.random.negative_binomial(Ir[j, 0], prob)
        seed_ij_1[j] = np.random.multinomial(delta_Ij[j], pvals)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
    for j in range(scale):
        for i in range(3):
            seed_ij_2[i][j] = np.random.multinomial(y[i][j], np.array(commuting_ratio[j]))[0:scale]

    fun_matrix = np.zeros((9, scale))
    fun_matrix[0] = np.sum(seed_ij_1, axis=0).reshape(1, scale)
    
    for i in range(3):
        fun_matrix[2*i+1] = np.sum(seed_ij_2[i], axis=1).reshape(1, scale)
        fun_matrix[2*i+2] = np.sum(seed_ij_2[i], axis=0).reshape(1, scale)
        
    fun_matrix[7] = np.random.poisson(y[1]/D)
    fun_matrix[8] = np.random.poisson(y[2]/L)

    num_SIR[0] = y[0] - fun_matrix[0] - fun_matrix[1] + fun_matrix[2] + fun_matrix[8]
    num_SIR[1] = y[1] + fun_matrix[0] - fun_matrix[3] + fun_matrix[4] - fun_matrix[7]
    num_SIR[2] = y[2] + fun_matrix[7] - fun_matrix[5] + fun_matrix[6] - fun_matrix[8]
    
    new_N = num_SIR[0] + num_SIR[1] + num_SIR[2]
    newcase = fun_matrix[0]

    # Handle negative values
    if np.min(num_SIR) < 0: 
        for i in range(scale):
            if np.min(num_SIR[:, i]) < 0:
                for j in range(3):
                    if num_SIR[j, i] < 0:
                        num_SIR[j, i] = 1
                sum_val = np.sum(num_SIR[:, i])
                weight = num_SIR[:, i]/sum_val
                num_SIR[:, i] = new_N[i] * weight
                
    return num_SIR, new_N, seed_ij_1, seed_ij_2[1], newcase

def get_standard(prior_sam_state, num_ens, uhf_id, N_all):
    """Standardize sample states"""
    post_sam_state = np.zeros((num_ens, 3))
    for i in range(num_ens):
        for j in range(3):
            if prior_sam_state[i, j] < 0:
                prior_sam_state[i, j] = 0
        sum_val = np.sum(prior_sam_state[i])
        weight = prior_sam_state[i]/sum_val
        post_sam_state[i] = N_all[i, uhf_id] * weight
    return post_sam_state

def get_modify(prior_state, ens_id, N_all):
    """Modify state values to be within bounds"""
    post_state = prior_state.copy()
    for l in range(len(ens_id)):
        i = int(ens_id[l])
        for k in range(scale):
            for j in range(3):
                if prior_state[i][j][k] < 0:
                    post_state[i][j][k] = random.uniform(0.1, 0.3) * np.mean(prior_state[:, j, k])
                if prior_state[i][j][k] > N_all[i, k]:
                    post_state[i][j][k] = random.uniform(0.8, 0.9) * N_all[i, k]
            
            sum_val = np.sum(post_state[i, :, k])
            weight = post_state[i, :, k]/sum_val
            post_state[i, :, k] = N_all[i, k] * weight  
    return post_state

def check_parameter_bounds(prior_data, flag, p0, num_ens, para0):
    """Ensure parameters stay within bounds"""
    post_data = prior_data.copy()
    if flag == 0:
        if post_data[prior_data > 0.6].size > 0 or post_data[prior_data < 0.2].size > 0:
            for i in range(num_ens):
                if prior_data[i] > 0.6:
                    post_data[i] = random.uniform(0.55, 0.6)
                if prior_data[i] < 0.2:
                    post_data[i] = random.uniform(0.2, 0.25)
    else:
        if post_data[prior_data > np.max(para0[:, flag])].size > 0 or post_data[prior_data < np.min(para0[:, flag])].size > 0:
            for i in range(num_ens):
                if prior_data[i] < np.min(para0[:, flag]):
                    post_data[i] = random.uniform(1, 1.5) * np.min(para0[:, flag])
                if prior_data[i] > np.max(para0[:, flag]):
                    post_data[i] = random.uniform(0.7, 0.9) * np.max(para0[:, flag])
    return post_data

def check_bounds(prior_data, flag, uhf_id, num_ens, N_all):
    """Ensure state values stay within bounds"""
    post_data = prior_data.copy()
    if flag == 0:
        if post_data[prior_data > N_all[:, uhf_id]].size > 0 or post_data[prior_data < 0].size > 0:
            for i in range(num_ens):
                if prior_data[i] > N_all[i, uhf_id]:
                    post_data[i] = 0.9 * N_all[i, uhf_id]
                if prior_data[i] < 0:
                    post_data[i] = random.uniform(0.1, 0.3) * np.mean(prior_data)
    else:    
        if post_data[prior_data > N_all[:, uhf_id]].size > 0 or post_data[prior_data < 0].size > 0:
             for i in range(num_ens):
                if prior_data[i] > N_all[i, uhf_id]:
                    post_data[i] = 0.5 * N_all[i, uhf_id]
                if prior_data[i] < 0:
                    post_data[i] = random.uniform(0.1, 0.3) * np.mean(prior_data)
    return post_data

def train_EAKF(t1, t2, obs_power, sam_0, para_0, num_ens, wnum, p0, matrix, part_commuting_month, scale, MSA_real_case):
    """Ensemble Adjustment Kalman Filter training"""
    T = t2 - t1
    state_sam_record = np.zeros((T, num_ens, 3, scale))
    para_record = np.zeros((wnum, num_ens, scale, len(p0)))
    state_sam_record[0] = sam_0
    para_record[0] = para_0
    post_case = np.zeros((wnum, scale, num_ens))
    new_N = np.zeros((T, num_ens, scale))
    new_case = np.zeros((T, num_ens, scale))
    day_time = np.zeros((T, num_ens, 2, scale, scale))
    new_N[0, :, :] = np.sum(sam_0, axis=1)
    
    for week in range(wnum - 1):
        truth = MSA_real_case[week]
        oevbase = 5**2
        obs_var = np.array(oevbase + truth**obs_power / 4)
        sim_case = np.zeros((7, num_ens, scale))
        index_t = 0
        
        for t in range(7):
            index_t = 7 * week + t
            commuting_month = part_commuting_month[index_t]
            for j in range(num_ens):
                beta_sam = para_record[week][j, :, 0]
                D_sam = para_0[j, 0, 2]
                L_sam = para_0[j, 0, 3]
                state_sam_record[index_t + 1][j], new_N[index_t + 1][j], day_time[index_t + 1][j][0], day_time[index_t + 1][j][1], new_case[index_t + 1][j] = SIR(
                    state_sam_record[index_t][j], new_N[index_t][j], beta_sam, L_sam, D_sam, matrix, commuting_month, scale)
            
            gamma = 0.2
            sim_case[t] = new_case[index_t + 1, :, :] / gamma

        week_case = np.sum(sim_case, axis=0)
        para_record[week + 1] = para_record[week]
        MSA_nonezero = []
        
        for te in range(scale):
            if np.max(MSA_real_case[:, te]) != 0:
                MSA_nonezero.append(te)
                
        for k in MSA_nonezero:
            if np.var(week_case[:, k]) == 0:
                prior_var = 0.1
                post_var = 0.1
            else:
                prior_var = np.var(week_case[:, k])
                post_var = prior_var * obs_var[k] / (prior_var + obs_var[k])
                
            prior_mean = np.mean(week_case[:, k])
            post_mean = post_var * (prior_mean / prior_var + truth[k] / obs_var[k])
            alpha = (obs_var[k] / (obs_var[k] + prior_var))**0.5
            delta = post_mean + alpha * (week_case[:, k] - prior_mean) - week_case[:, k]
            post_case[week][k] = week_case[:, k] + delta
            
            for i in range(3):
                corr = (np.cov([state_sam_record[index_t + 1, :, i, k], week_case[:, k]])[0][1]) / prior_var
                state_sam_record[index_t + 1, :, i, k] += corr * delta
                state_sam_record[index_t + 1, :, i, k] = check_bounds(state_sam_record[index_t + 1, :, i, k], i, k, num_ens, new_N[index_t + 1]) 
                
            state_sam_record[index_t + 1, :, :, k] = get_standard(state_sam_record[index_t + 1, :, :, k], num_ens, k, new_N[index_t + 1])  
            
            for i in range(1):
                corr = (np.cov([para_record[week + 1, :, k, i], week_case[:, k]])[0][1]) / prior_var
                para_record[week + 1, :, k, i] += corr * delta
                para_record[week + 1, :, k, i] = check_parameter_bounds(para_record[week + 1, :, k, i], i, p0, num_ens, para_0)

    predic_state = np.array([state_sam_record[7 + 7 * k1] for k1 in range(wnum - 1)])
    predic_para = para_record[1:]
    return predic_para, predic_state, post_case, new_N, day_time

def getParaStart(num_ens, scale, p0):
    """Initialize parameters"""
    paraStart = np.zeros((num_ens, scale, len(p0)))
    for i in range(num_ens):
        D_sam = np.random.randint(1.9, 6)
        L_sam = np.random.randint(2 * 365, 10 * 365)
        for j in range(scale):
            paraStart[i, j, 0] = np.random.uniform(0.2, 0.6)
            paraStart[i, j, 1] = np.random.uniform(0.05, 0.3)
            paraStart[i, j, 2] = D_sam
            paraStart[i, j, 3] = L_sam
    return paraStart

def getStart(num_ens, seed_loc, N_all, scale, seed_ratio):
    """Initialize state values"""
    StateStart = np.zeros((num_ens, 3, scale))
    for i in range(num_ens):
        StateStart[i][0] = N_all
        sum_people = np.random.randint(500, 1000) * len(seed_loc)
        for j in seed_loc:
            StateStart[i][1][j] = sum_people * seed_ratio[j]
            StateStart[i][2][j] = 0
            StateStart[i][0][j] = N_all[j] - StateStart[i][1][j] - StateStart[i][2][j]
    return StateStart  

def get_key_from_value(d, values):
    """Get dictionary key from value"""
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        found_key = None
        for key, val in d.items():
            if val == value:
                found_key = key
                break
        result.append(found_key)
    return result

def find_may_next_points(seed_set, no_edge_point, part_commuting_matrix, daily_matrix, 
                        wnum, seed_threshold, MSA_220_list):
    may_next_point = []
    for j in seed_set:
        for y in no_edge_point:
            if (part_commuting_matrix[j, y] != 0 or part_commuting_matrix[y, j] != 0):
                if y not in may_next_point and y in MSA_220_list:
                    if wnum >= seed_threshold[y]:
                        may_next_point.append(y)
            if (np.sum(daily_matrix[:int((wnum+1)*7), j, y], axis=0) != 0 or 
                np.sum(daily_matrix[:int((wnum+1)*7), y, j], axis=0) != 0):
                if y not in may_next_point and y in MSA_220_list:
                    if wnum >= seed_threshold[y]:
                        may_next_point.append(y)
    set1 = set(may_next_point)
    sorted_list1 = [item for item in no_edge_point if item in set1] 
    return sorted_list1

def calculate_parallel_mae(
        initial_state, 
        initial_parameters, 
        initial_para0,
        target_j,
        commuting_matrix,
        airport_matrix,
        current_time,
        target_i,
        partial_commuting_matrix,
        partial_commuting_month,
        location_list,
        last_week_real,
        MSA_real_case):
    week_number = current_time+1
    start_time = time.time()
    np.random.seed(int(start_time + random.randint(1, 1000) + target_j))
    location_map = {index: value for index, value in enumerate(location_list)}
    working_commuting_matrix = commuting_matrix.copy()
    working_airport_matrix = airport_matrix.copy()
    working_commuting_matrix[target_i, target_j] = partial_commuting_matrix[target_i, target_j]
    working_commuting_matrix[target_j, target_i] = partial_commuting_matrix[target_j, target_i]
    working_airport_matrix[:, target_i, target_j] = partial_commuting_month[:, target_i, target_j]
    working_airport_matrix[:, target_j, target_i] = partial_commuting_month[:, target_j, target_i]
    working_commuting_matrix = working_commuting_matrix[location_list, :][:, location_list]
    working_airport_matrix = working_airport_matrix[:, location_list, :][:, :, location_list]

    num_days = week_number * 7
    seed_matrix = np.zeros((num_days, 2, len(location_list), len(location_list)))
    target_key = get_key_from_value(location_map, target_i)
    num_locations = len(location_list)
    num_samples = 100
    forecast_horizon = 7 + 1
    trained_params, trained_state, trained_cases, new_population, daily_results = train_EAKF(
        0, week_number*7, 2, initial_state, initial_para0, num_samples, 
        week_number, initial_parameters, working_commuting_matrix, 
        working_airport_matrix, num_locations, MSA_real_case[:, location_list]
    )
    forecast_state = np.zeros((forecast_horizon, num_samples, 3, num_locations))
    forecast_population = np.zeros((forecast_horizon, num_samples, num_locations))
    forecast_cases = np.zeros((forecast_horizon, num_samples, num_locations))
    forecast_daily = np.zeros((forecast_horizon, num_samples, 2, num_locations, num_locations))
    forecast_state[0] = trained_state[-1]
    forecast_population[0, :, :] = np.sum(forecast_state[0], axis=1)

    for time_step in range(forecast_horizon - 1):
        current_month_commuting = working_airport_matrix[time_step, :, :]
        for sample in range(num_samples):
            beta = trained_params[-1:, sample, :, 0][0]
            D = initial_para0[sample, 0, 2]
            L = initial_para0[sample, 0, 3]
            (forecast_state[time_step + 1][sample],
             forecast_population[time_step + 1][sample],
             forecast_daily[time_step + 1][sample][0],
             forecast_daily[time_step + 1][sample][1],
             forecast_cases[time_step + 1][sample]) = SIR(
                forecast_state[time_step][sample],
                forecast_population[time_step][sample],
                beta, L, D,
                working_commuting_matrix,
                current_month_commuting,
                num_locations
            )
    gamma = 0.2 
    last_week_samples = np.sum(forecast_cases[-7:] / gamma, axis=0)[:, target_key[0]]
    last_week_forecast = abs(np.median(last_week_samples) - MSA_real_case[current_time, target_i])
    mae = last_week_real - last_week_forecast
    seed_matrix[:-7] = np.mean(daily_results[:-7], axis=1)
    seed_matrix[-7:] = np.mean(forecast_daily[1:], axis=1)

    return target_j, mae, last_week_samples, np.sum(seed_matrix, axis=0)

def max_loc(truth):
    location_list = []
    week = 2
    for i in range(len(truth)-week):
        if truth[i+1] < truth[i]  and truth[i+2] < truth[i] and truth[i]>10:
            location_list.append(i)
    if len(location_list) == 0:
        part_truth = [truth[-1],truth[-2],truth[-3],truth[-4],truth[-5]]
        max_loc =  np.argmax(part_truth)
        location_list.append(int(len(truth)-1-max_loc))
    return location_list
def over_1000(data):
    for index, value in enumerate(data):
        if value >= 20000:
            return index
    return 100
def optimize_parameters(truth, T_j, week_num):
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
        loc_week = truth[:, i]
        if np.max(loc_week) > 0:
            break_1000 = over_1000(loc_week)
            if break_1000 != 100:
                loc = min(np.min(max_loc(loc_week)),break_1000)
            else:
                loc = np.min(max_loc(loc_week))
            truth_max.append(loc)
        else:
            timelist.append(100)
            betalist.append([0, 0, 0])
            truth_max.append(0)
            continue
        
        if loc < week_num:
            week = loc
        else:
            week = week_num
            
        L1 = []
        paralist = []
        for t_j in T_j:
            x = np.arange( loc - week, loc + 1)
            y = truth[:, i][ loc - week : loc + 1]
            x0 = (0.2, 0.2, 0.2)
            bounds = [(None, None), (1e-5, None), (1e-5, None)] 
            res = minimize(fun, x0, args=(x, y, t_j), method='Nelder-Mead', bounds=bounds)
            paralist.append(res.x)
            L1.append(-res.fun)
        
        max_point = np.argmax(L1)
        timelist.append(T_j[max_point])
        betalist.append(paralist[max_point])
    
    return timelist, betalist, truth_max

# Load data
commuting_matrix = pd.read_csv("commuting_matrix.csv").values
data = np.load('daily_flight.npz')
daily_matrix = data['daily_matrix']
MSA_real_case = np.round(pd.read_csv("ILI+_incidence.csv").values*100000)
T_j = list(np.arange(0,MSA_real_case.shape[0],1/14))
onset_time, betalist, truth_max = optimize_parameters(MSA_real_case, T_j, MSA_real_case.shape[0])
timelist = pd.DataFrame(np.zeros((369,2)))
timelist.iloc[:,0] = index_list = [int(i) for i in range(369)]
timelist.iloc[:,1] = onset_time
timelist = timelist.sort_values(by=timelist.columns[1], ascending=True)
seed_threshold=np.floor(onset_time).reshape(scale,1)
seed_list = [41, 302, 211]
MSA_220_list = []
for i in range(scale):
    if np.max(MSA_real_case[:, i]) != 0:
        MSA_220_list.append(i)

# Initialize seed ratios
y = []
for i in range(scale):
    if i in seed_list:
        if i == 41:
            y.append(10)
        elif i == 211:
            y.append(45)
        elif i == 302:
            y.append(6)
        else:
            y.append(np.random.randint(1, 5))
    else:
        y.append(0)
seed_ratio = y / np.sum(y)

# Initialize sets and matrices
seed_set = seed_list.copy()
wnum = 2
seed_location = seed_list.copy()
no_edge_point = []
for i in timelist.iloc[:, 0].values:
    if i not in no_edge_point and i in MSA_220_list and i not in seed_set:
        no_edge_point.append(int(i))

# Initialize infection matrices
inf_commuting_matrix = np.zeros((scale, scale))
inf_airport_matrix = np.zeros((daily_matrix.shape[0], scale, scale))
inf_contact_matrix = np.zeros((scale, scale))
inf_path_transsmission = np.zeros((scale, scale))
edge_order = 0

# Initialize seed connections
for i in seed_set:
    for j in seed_set:
        inf_commuting_matrix[i, j] = commuting_matrix[i, j]
        inf_commuting_matrix[j, i] = commuting_matrix[j, i]
        inf_airport_matrix[:, i, j] = daily_matrix[:, i, j]
        inf_airport_matrix[:, j, i] = daily_matrix[:, j, i]

# Set diagonal of commuting matrix
for i in range(scale):
    inf_commuting_matrix[i, i] = commuting_matrix[i, i]

# Main loop
while len(no_edge_point) > 0 and wnum <= MSA_real_case.shape[0]:
    may_next_point = find_may_next_points(
        seed_set, no_edge_point, commuting_matrix, 
        daily_matrix, wnum, seed_threshold, MSA_220_list
    )
    error_point = []
    if len(may_next_point) == 0:
        wnum += 1
        continue  
    while len(may_next_point) > 0:
        i = may_next_point[0]
        flag_edge = 0
        inf_seed_set = []
        for y in seed_set:
            if commuting_matrix[i, y] != 0 or commuting_matrix[y, i] != 0:
                inf_seed_set.append(y)
            if np.sum(daily_matrix[:int((wnum + 1) * 7), i, y], axis=0) != 0 or \
               np.sum(daily_matrix[:int((wnum + 1) * 7), y, i], axis=0) != 0:
                if y not in inf_seed_set:
                    inf_seed_set.append(y)
        # Initialize matrices for analysis
        error_matrix = np.zeros((scale, sam_num))
        MAE_matrix = np.zeros((scale, scale))
        choose_list = seed_set.copy()
        
        if i not in choose_list:
            choose_list.append(i)
        choose_list = sorted(choose_list)
        
        # Prepare indexed numbers and N0
        indexed_numbers = {index: value for index, value in enumerate(choose_list)}
        N0 = np.sum(commuting_matrix, axis=1)
        choose_N0 = [N0[number] for number in choose_list]
        small_scale = len(choose_list)
        
        # Get initial state and parameters
        state0 = getStart(
            sam_num, 
            get_key_from_value(indexed_numbers, seed_location), 
            choose_N0, 
            small_scale,
            seed_ratio[choose_list]
        )
        p0 = np.array([0.3, 0.2, 5, 730])
        para0 = getParaStart(sam_num, small_scale, p0)
        
        # Initialize matrices
        seed_matrix = np.zeros((scale, 2, len(choose_list), len(choose_list)))
        last_week_fore = abs(MSA_real_case[wnum, i])
        
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=35) as executor:
            futures = {
                executor.submit(
                    calculate_parallel_mae, 
                    state0, p0, para0, j, 
                    inf_commuting_matrix, 
                    inf_airport_matrix, 
                    wnum, i, 
                    commuting_matrix, 
                    daily_matrix,  
                    choose_list, 
                    last_week_fore, 
                    MSA_real_case
                ): j for j in inf_seed_set
            }
            
            for future in concurrent.futures.as_completed(futures):
                j, MAE, last_new_case, seed_commute_fly = future.result()
                MAE_matrix[j, i] = MAE
                error_matrix[j] = last_new_case
                seed_matrix[j] = seed_commute_fly

        if np.max(MAE_matrix) > 0:
            max_loc = np.max(MAE_matrix[:, i])
            for seed in range(scale):
                if MAE_matrix[seed, i] == max_loc:
                    sample1 = abs(np.zeros((sam_num, 1)) - MSA_real_case[wnum, i]).flatten().tolist()
                    sample2 = abs(error_matrix[seed] - MSA_real_case[wnum, i]).flatten().tolist()
                    stat, p_value = st.mannwhitneyu(sample1, sample2, alternative='greater')
                    if p_value < 0.05:
                        flag_edge = 1
                        inf_contact_matrix[seed, i] = wnum
                        seed_idx = get_key_from_value(indexed_numbers, seed)[0]
                        i_idx = get_key_from_value(indexed_numbers, i)[0]  
                        if seed_matrix[seed, 0, seed_idx, i_idx] > seed_matrix[seed, 1, seed_idx, i_idx]:
                            inf_path_transsmission[seed, i] = 1
                        else:
                            inf_path_transsmission[seed, i] = 2   
                        inf_commuting_matrix[seed, i] = commuting_matrix[seed, i]
                        inf_commuting_matrix[i, seed] = commuting_matrix[i, seed]
                        inf_airport_matrix[:, seed, i] = daily_matrix[:, seed, i]
                        inf_airport_matrix[:, i, seed] = daily_matrix[:, i, seed]
                        inf_seed_set.remove(seed)
                        
        # Handle cases where no edge was found
        if flag_edge == 0:
            may_next_point.remove(i)
            error_point.append(i)
            
            if seed_threshold[i] + 3 <= wnum:
                seed_set.append(i)
                no_edge_point.remove(i)
                
                if seed_threshold[i] <= 2:
                    seed_location.append(i)
                    may_next_point = find_may_next_points(
                        seed_set, no_edge_point, commuting_matrix, 
                        daily_matrix, wnum, seed_threshold, MSA_220_list
                    )
                    for j in error_point:
                        if j in may_next_point:
                            may_next_point.remove(j)
            continue
            
        # Handle successful edge connection
        if flag_edge == 1:
            seed_set.append(i)
            no_edge_point.remove(i)
            may_next_point = find_may_next_points(
                seed_set, no_edge_point, commuting_matrix, 
                daily_matrix, wnum, seed_threshold, MSA_220_list
            )
            for j in error_point:
                if j in may_next_point:
                    may_next_point.remove(j)         
    if len(no_edge_point) == 0:
        break
        
    # Prepare for next week
    wnum += 1
    true_week = wnum
    no_edge_point = []
    for i in timelist.iloc[:, 0].values:
        if i not in no_edge_point and i in MSA_220_list and i not in seed_set:
            no_edge_point.append(int(i))
    pd.DataFrame(inf_contact_matrix).to_csv("inf_contact_matrix_"+str(result_id)+".csv", index=None)
    pd.DataFrame(inf_path_transsmission).to_csv("transsmission_matrix_"+str(result_id)+".csv", index=None)