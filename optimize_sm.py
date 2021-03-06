
# coding: utf-8

# Math 485 Spring 2019 Prof Lega
# Supermodel Group
#
# Created by: Alex Stoken 18 Feb 2019
#
# Last updated: 26 Feb 2019
#
# This file takes input data and returns the minimum connection coefficient vector C

# # TODO
# * [DONE] make (F,C) output a csv so that it's ready to be read in again
# * add docstrings
# * [DONE] increase efficiency
# * [DONE] wrap the animation routine in a function
# * [DONE] use a random C vector to start to find other local minima
# * write a routine to increase K over time
# * [DONE] Make (SM - Truth) plot for each direction
# * Check how t is calculated with the gamma element

# In[1]:


import numpy as np
import pandas as pd
import scipy.optimize as optimize #has the conjugate gradient method
import scipy.integrate as integrate #ODE solver
from IPython.display import clear_output

import datetime #for reference
import argparse


def load_lorentz_data(fname):
    """
    loads data from xlsx file, returns np.array()
    """
    data = pd.read_excel(fname, names=['x','y','z'], header = None)
    return np.array(data)


def get_lastest_C(fname):
    """
    returns last C vect from input file
    """
    C_headers = ['F','cx12', 'cx13', 'cx23', 'cx21', 'cx31', 'cx32','cy12', 'cy13', 'cy23', 'cy21', 'cy31', 'cy32', 'cz12', 'cz13', 'cz23', 'cz21', 'cz31', 'cz32']
    F_C_df = pd.read_csv(fname, header = None,index_col = False, names = C_headers)
    return np.array(F_C_df.iloc[-1])[1:]


def cost_fn(C, K =200, gamma= 0.4, delta =1, fname= None, data = None, verbose = True):
    """
    calculate value of cost fn from paper for given model

    parameters:
    C - vector of conn coefficients, this is the indep var
    K - number of points to train on/calculate cost fn over
    gamma - disocunt factor to reduce importance of future error
    delta - length of time to integrate over
    fname - file to write each (F,C) pair to
    data - true values of the lorenz63 model
    verbose - if true, prints output
    """

    clear_output(wait = True)
    #time from 0 to 20 seconds, with a point taken every 0.01 s
    t = np.arange(0,20,0.01)

    #take the integral of the lorentz system with the given Cs
    sum_k = 0

    #set x_o,y_o, x_o
    x_init = [x_o[0,i] for i in range(3) for i in range(3)]

    #do the summation in the cost function over all K
    for i in range(K):
        #sum_k = integrate.quad(integrand,t[i], t[i]+delta,args = (data, C, t,i,0.4))[0]

        #each K we want to integrate the ODE w/ connection coefficiants
        sol = integrate.odeint(conn_lorentz, x_init, t, args=(C,[13.25,7,6.5], [19,18,38],[3.5, 3.7, 1.7]))

        s_model = supermodel(sol)

        #set this value to 0, so we can start the addition ourselves
        integrand = 0

        for j in range(100):
            #range is from t_i to t_i + delta, but delta is 1 full second, so 100 of our timesteps
            model = s_model[i + j]
            truth = data[i+j,0:3]
            diff = model - truth

            #paper suggestion
            integrand += np.linalg.norm(diff)**2 * (gamma**t[i+j])

            #if we want t - t_i (Lega suggestion)
            #integrand += np.linalg.norm(diff)**2 * (gamma**(t[i+j]- t[i]))

        sum_k += integrand

    #compute normalization constant

    norm = 1/ (K * delta)

    #set F to be returned
    F = sum_k * norm

    #write C vector to file for later access
    if fname != None:
        with open(fname,'a') as f:
            C_str = str(C.tolist())
            C_to_file = C_str.strip('[]')
            f.write(str(F) + ',' + C_to_file +',' + '\n')

    #print out current F and C vec
    if verbose == True: print('\r', F,C , end='')

    #return cost F
    return F

def conn_lorentz(x_vect, t0,C, sig = [10,10,10], rho = [28,28,28], beta = [8 /3,8/3,8/3]):
    """
    Parameters:
    sigma, rho, beta - model parameters from climate data, defaults to all true values


    Returns:
    derivs vector for next timestep
    """
    x1,y1,z1, x2, y2, z2, x3, y3, z3 = x_vect

    cx12, cx13, cx23, cx21, cx31, cx32,     cy12, cy13, cy23, cy21, cy31, cy32,     cz12, cz13, cz23, cz21, cz31, cz32 = C

    C_x1 = cx12 * (x2 - x1) + cx13 * (x3-x1)
    C_x2 = cx21 * (x1 - x2) + cx23 * (x3-x2)
    C_x3 = cx31 * (x1 - x3) + cx32 * (x2-x3)

    C_y1 = cy12 * (y2 - y1) + cy13 * (y3-y1)
    C_y2 = cy21 * (y1 - y2) + cy23 * (y3-y2)
    C_y3 = cy31 * (y1 - y3) + cy32 * (y2-y3)

    C_z1 = cz12 * (z2 - z1) + cz13 * (z3-z1)
    C_z2 = cz21 * (z1 - z2) + cz23 * (z3-z2)
    C_z3 = cz31 * (z1 - z3) + cz32 * (z2-z3)

    sig1,sig2,sig3 = sig
    rho1,rho2,rho3 = rho
    beta1,beta2,beta3 = beta

    dx1 = sig1 * (y1-x1) + C_x1
    dy1 = x1 * (rho1 - z1) -y1 + C_y1
    dz1 = x1 * y1 - beta1 *z1 +C_z1

    dx2 = sig2 * (y2-x2) + C_x2
    dy2 = x2 * (rho2  -z2) - y2 + C_y2
    dz2 = x2 * y2 - beta2 *z2 +C_z2

    dx3 = sig3 * (y3-x3) + C_x3
    dy3 = x3 * (rho3 - z3) - y3 + C_y3
    dz3 = x3 * y3 - beta3 *z3 +C_z3


    return [dx1,dy1,dz1,dx2,dy2,dz2,dx3,dy3,dz3]

def supermodel(sol):
    """
    input is three 3-vectors of model solns

    returns 3 vector of supermodel solns
    """
    x_s = 1/3 * (sol[:,0:3] + sol[:,3:6] + sol[:,6:9])
    return x_s


def run_sm_optimization(last_fname = None, verbose = True, k = 10):
    """
    parameters:
    last_fname - name of the file from the last run of the algorithm

    return:
    optimal C vec
    """
    #set parameter values
    #taken from paper
    gamma = 0.4
    delta = 1
    K = k


    #load data and set initial conditions
    x_o = load_lorentz_data('TrueValues_SuperModel.xlsx')
    x_init = [x_o[0,i] for i in range(3) for i in range(3)]
    C_init = np.zeros(18)
    t = np.arange(0,20,0.01)


    C_headers = ['F','cx12', 'cx13', 'cx23', 'cx21', 'cx31', 'cx32','cy12', 'cy13', 'cy23', 'cy21', 'cy31', 'cy32', 'cz12', 'cz13', 'cz23', 'cz21', 'cz31', 'cz32']

    #if given a last file to restart from, tell gradient method that this is our guess
    #else make random guess between in [10,10] for each C component
    if last_fname != None:
        C_opt_guess = get_lastest_C(last_fname)
    else:
        C_opt_guess = 10*np.random.random(18)


    #find true values for comparison
    truth = integrate.odeint(conn_lorentz, x_init, t, args=(C_init,[10,10,10], [28,28,28],[8/3, 8/3, 8/3]))
    data = truth[:, 0:3]


    #make file to print (F,C) pairs to
    fname = 'F_list_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.csv'
    f = open(fname, 'w+')
    f.close()

    #begin optimizatio
    print('SM optimization for K= %s beginning with C = %s' %(str(K),str(C_opt_guess)))

    #
    C = optimize.fmin_cg(cost_fn, C_opt_guess, args=(K,gamma,delta, fname, data, verbose), full_output=True)


    print('SM has been optimized')
    print(C)

    return C


# # --------------------- RUN OPTIMIZATION ------------------------

# # List of which files are which:
#
# * K:10 time: t0 paper File: 'F_list_2019-03-24_10:58:12.csv' SM1 routine finished
# * K:10 time: t0 File: 'F_list_2019-03-16_22:02:39.csv' SM2 routine finished
# * K:10 time: t - ti File: 'F_list_2019-03-26_09:32:42.csv' SM3 routine finished
# * K:10 time: t - ti File:
#


parser = argparse.ArgumentParser()

parser.add_argument('--k', type = int, help = 'K initializations for cost fn')

parser.add_argument('--last_file', type = str, help = '')

parser.add_argument('--verbose', action='store_true', help = '')

args = parser.parse_args()


#to run from last place left off
x_o = load_lorentz_data('TrueValues_SuperModel.xlsx')
if args.last_file != None:

    C_opt = run_sm_optimization(args.last_file, verbose = args.verbose, k =args.k)

#to run a new version
else:
    C_opt = run_sm_optimization()
