from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:39:25 2020

@author: Hussain Kharodawala
"""

"""

Implementing Set Covering Problem using various techniques:
- MIP Model using pyomo.
- MIP Model using Google OR-tools.
- Using Lagrange Relaxation and Lagrangean Heuristic to get a good feasible solution.
References:
    
- Integer programming solution methods - J E Beasley - http://people.brunel.ac.uk/~mastjjb/jeb/natcor_ip_rest.pdf)
- Lagrangian relaxation can solve your optimization problem much, much faster - https://medium.com/bcggamma/lagrangian-relaxation-can-solve-your-optimization-problem-much-much-faster-daa9edc47cc9)
- Geoffrion, Arthur M. "Lagrangian relaxation for integer programming." 50 Years of Integer Programming 1958-2008. Springer, Berlin, Heidelberg, 2010. 243-281.
- Fisher, Marshall L. "The Lagrangian relaxation method for solving integer programming problems." Management science 27.1 (1981): 1-18.
- Beasley, John E. "A lagrangian heuristic for set‐covering problems." Naval Research Logistics (NRL) 37.1 (1990): 151-164. 

Datasets taken from: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
"""

#import packages
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

def get_data():
    data = {}
    data['m'] = 3 #Number of Constraints / Rows
    data['n'] = 4 #Number of Variables / Columns
    data['items'] = np.array(range(data['n'])) 
    data['rows_index'] = np.array(range(data['m']))
    data['costs'] = np.array([2,3,4,5]) #Costs of variables
    data['rows'] = np.array([[0,2],[0,3],[1,2,3]]) #Columns covering each row
    
    data['t'] = [] #Stores the cost of the variable with the least cost associated with the constraints
    data['minpos'] = [] #Stores the index of the variable with the least cost associated with the constraints
    for i in data['rows_index']:
        l = dict((j,data['costs'][j]) for j in data['rows'][i])
        minpos_ = [key for key in l if all(l[temp] >= l[key] for temp in l)]
        data['t'].append(data['costs'][minpos_[0]])
        data['minpos'].append(minpos_[0])
        
    data['rows_covered'] = [[] for j in data['items']]
    for i in data['rows_index']:
        for j in data['rows'][i]:
            data['rows_covered'][j].append(i)
    data['rows_covered'] = np.array(data['rows_covered'],dtype=object)
    
    return data

def read_data(file_loc):
    file_location = fr"{file_loc}".strip()
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    data = {}
    lines = input_data.split('\n')
    count = 0
    m,n = lines[count].split()
    count+=1
    data['m'] = int(m)
    data['n'] = int(n)
    data['items'] = np.array(range(data['n']))
    data['rows_index'] = np.array(range(data['m']))
    data['costs'] = []
    while len(data['costs']) < data['n']:
        parts = lines[count].split()
        count+=1
        data['costs'] += parts
    data['costs'] = np.array(list(map(float, data['costs'])))
    
    data['rows'] = []
    while len(data['rows']) < data['m']:
        cols = int(lines[count].split()[0])
    #     print(cols)
        count+=1
        i = 0
        row = []
        while len(row) < cols:
            row += lines[count].split()
            count+=1
        data['rows'].append(row)
    for i in range(len(data['rows'])):
        data['rows'][i] = np.array(list(map(int, data['rows'][i])))
        data['rows'][i] = np.array(list(map(lambda x: x - 1, data['rows'][i])))
    
    data['t'] = [] #Stores the cost of the variable with the least cost associated with the constraints
    data['minpos'] = [] #Stores the index of the variable with the least cost associated with the constraints
    for i in data['rows_index']:
        l = dict((j,data['costs'][j]) for j in data['rows'][i])
        minpos_ = [key for key in l if all(l[temp] >= l[key] for temp in l)]
        data['t'].append(data['costs'][minpos_[0]])
        data['minpos'].append(minpos_[0])
        
    data['rows_covered'] = [[] for j in data['items']]
    for i in data['rows_index']:
        for j in data['rows'][i]:
            data['rows_covered'][j].append(i)
    data['rows_covered'] = np.array(data['rows_covered'],dtype=object)
    
    return data

def mip_pyomo(data,time_limit:"Solver Time Limit (s)" = 1800,verbose=False) -> "SCP MIP using pyomo":
    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    
    import pyomo
    import pyomo.environ as pe
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
    
    scp = pe.ConcreteModel()

    scp.x = pe.Var(data['items'],domain = pe.Binary)
    
    scp.obj = pe.Objective(expr = sum(data['costs'][j] * scp.x[j] for j in data['items']), sense=pe.minimize)
    
    def rf_c(scp,i):
        row = data['rows'][i]
        return sum(scp.x[j] for j in row) >= 1
    scp.constraints = pe.Constraint(data['rows_index'],rule = rf_c)
    
    solver = SolverFactory('cbc',executable=r"C:\Cbc-2.10-win32-msvc14\bin\cbc",options={'sec':time_limit})
    result = solver.solve(scp,tee=verbose)
    
    #If the output id Optimal or Feasible
    if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
        print("Status: ",result.solver.termination_condition)
        print('Objective value =', scp.obj.expr())
        print()
        verboseprint(result.solver)
    #In case it is not feasible or optimal
    else:
        print('The solver did not find an optimal solution.')
        verboseprint(result.solver)
        print()
    try:
        objValue = scp.obj.expr()
        out = {}
        for j in data['items']:
            if scp.x[j].value == 1:
                out[j] = 1

    except:
        objValue = None
        out =None
        print("Something went wrong! Model didn't exit with a valid solution.")
        print()
    
    verboseprint(f"Variables Values: {out}")
    return  objValue, out

def mip_OR_Tools(data,time_limit:"Solver Time Limit (s)" = 1800,verbose:"Long Output"=False) -> "SCP MIP using OR-tools":
    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    
    #import packages
    from ortools.linear_solver import pywraplp 

    solver = pywraplp.Solver("SCP",pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) #Declare Model
    solver.EnableOutput() #Enable Log Printing (May not work)
    solver.set_time_limit(time_limit*1000) #Set Time Limit for the Solver Run
    
    x = {} #Variable Dict
    for j in data['items']:
        x[j] = solver.NumVar(0,1,'x[%i]' %j) #Variable Declaration
    
    constraints = {} #Constraints Dict
    for i in data['rows_index']:
        constraints[i] = solver.Add(sum(x[j] for j in data['rows'][i]) >= 1) #Constraints Declaration
        
    solver.Minimize(solver.Sum(data['costs'][j] * x[j] for j in data['items'])) #Objective Function Declaration
    
    dict_status = {0:"Optimal", 1:"Feasible"} #OR-tools solver status dict 
    
    status = solver.Solve() #Solve the model
    
    #If the output id Optimal or Feasible
    if status == pywraplp.Solver.FEASIBLE or status == pywraplp.Solver.OPTIMAL:
        print("Status: ",dict_status[status])
        print('Objective value =', solver.Objective().Value())
        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
        print()
    #In case it is not feasible or optimal
    else:
        print('The problem does not have an optimal solution.')
        print()
    
    #Get the objective and variables values
    try:
        objValue = solver.Objective().Value()
        out = {}
        for j in data['items']:
            if x[j].solution_value() == 1:
                out[j] = 1
    #In case the model exited abnormally
    except:
        objValue = None
        out = None
        print("Something went wrong! Model didn't exit with a valid solution.")
        print()
    
    return objValue, out

def solve_llbp(data,lm:"Lagrange Multipliers",verbose=False):
    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    C = np.array([data['costs'][j] - sum(lm[i] for i in data['rows_covered'][j]) for j in data['items']])
    verboseprint("LLBP costs: ",C)
    verboseprint("Lagrange Multipliers: ",lm)
    x_array_llbp = (C<=0).astype(int)
    lb = np.nan_to_num(x_array_llbp*C).sum() + np.nan_to_num(lm).sum()
    verboseprint("LLBP Onjective: ", lb)
    verboseprint("LLBP solution: ",x_array_llbp)
    return lb , x_array_llbp, C

def heuristic(data,x_array_lm,verbose=False):
    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    x_local = x_array_lm.copy()
    for i in data['rows_index']:
        if sum(x_local[j] for j in data['rows'][i]) == 0:
            x_local[data['minpos'][i]] = 1
    x_ones = -np.sort(-np.array(np.where(x_local==1)[0]))
    verboseprint("Indices of variables that are 1 in LLBP Solution: ",x_ones)
    for j in x_ones:
        isImp = False
        for i in data['rows_covered'][j]:
            if sum(x_local[k] for k in data['rows'][i]) - x_local[j] == 0:
                isImp = True
                break
        if not isImp:
            x_local[j] = 0
    ub = np.nan_to_num(data['costs'] * x_local).sum()
    verboseprint("Bound from the heuristic: ",ub)
    verboseprint("Variable Values from the Heuristic: ",x_local)
    return ub , x_local

def LR_method(data,lmi,numIterations = 1000, verbose=False, plotResult=False):
    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    start  = pd.Timestamp('now') # Save the time-stamp

    lm =copy.copy(lmi) # Copy the initial Lagrange-Multipliers
    verboseprint(f"Initial multipliers: {lm}")
    verboseprint("*"*100)
    
    pi_ = 2 # Intitial value of pi which is a scaling factor for the step-size
    ub_ = math.inf # A placeholder for Loop Upper Bounds as +Infinity
    z_ub = math.inf # Set the initial Upper Bound as -Infinity 
    z_lb = -math.inf # Set the initial Lower Bound as -Infinity
    P = data['costs'].copy() # Variable Penalties as their costs initially
    originalCosts = data['costs'].copy() # Back-up the costs of the data
    
    ic = 0  # Iteration Counter
    gaps = [] # % gap between the best lower bound and best upper bound at each iteration
    llbpObjs = [] # Lower Bounds of each iteration are stored here
    lhObjs = [] # Upper Bounds of each iteration are stored here
    lbs = [] # Best Lower Bounds of each iteration are stored here
    ubs = [] # Best Lower Bounds of each iteration are stored here
    steps = 30 #Compare best LBs of the last "steps" iterations every iteration. If there is no improvement , then set: pi_ := pi_/2
    while ic<=numIterations:
        print(f"Iteration {ic}")
        print("-"*100)
        llbp_obj, x_array_lm, C = solve_llbp(data,lm,verbose=verbose) #Solve LLBP - Lagrange Lower Bound Problem
        z_lb = max(llbp_obj,z_lb) # Update the best lower bound
        llbpObjs.append(llbp_obj)
        lbs.append(z_lb) # Store the best lower bound for the iteration
        if ic%10 == 0 or lbs[-1] > lbs[-2]:
            """
            If the iteration count is a multiple of 50 OR we find an improvement in the lower bound, then
            Run the Lagrange Heuristic to generate a feasible Solution from the LLBP solution.
            """
            ub_ , x_array_h_= heuristic(data,x_array_lm,verbose=verbose) # Run the Lagrangean Heuristic
            # Update the best upper bound
            if ub_ < z_ub:
                z_ub = ub_
                best_solution = x_array_h_
        lhObjs.append(ub_)
        ubs.append(z_ub) # Store the best upper bound for the iteration
        print(f"LR heuristic objective at iteration {ic}: {round(ub_,2)}")
        print(f"Upper Bound at iteration {ic}: {round(z_ub,2)}")
        print()
        print(f"LLBP Solution at iteration {ic}: {round(llbp_obj,2)}")
        print(f"Lower bound at iteration {ic}: {round(z_lb,2)}")
        gap = (z_ub - z_lb) * 100 / z_ub # Calculate the gap between best UB and best LB.
        gaps.append(gap) # Store the gap
        print("Gap: ",gap) 
        if gap <= 2:
            # Exit the Method if gap is less than 2%
            break
        """
        Update the penalties of each variable.
        If the variable is selected in LLBP solution, Penalty = max(LLBP Objective, Curr_Penalty)
        If the variables is not selected in LLBP solution, Penalty = max(LLBP Objective + LLBP Co-efficient, Curr_Penalty)
        """
        P = np.array([max(P[j], llbp_obj + C[j]) if x_array_lm[j] == 0 else max(P[j], llbp_obj) for j in data['items']])
        verboseprint(f"Penalty (P) for iteration {ic}: {P}")
        """
        Update the costs of variables.
        If the Penalty of a variable is more than the best known upper bound,
        then the variable can be removed by setting its cost as infinity
        """
        data['costs'] = np.array([np.inf if P[j] > z_ub else data['costs'][j] for j in data['items']])
        verboseprint(f"Costs after iteration {ic}: {data['costs']}")
        # If No improvement in best Lower Bound last "steps" iteration, set pi_ := pi_/2
        if ic > steps:
            if all(ele == lbs[-30] for ele in lbs[-30:]):
                steps = ic+30
                pi_ = pi_/2
        print(f"pi_ for iteration {ic}: {pi_}")
        # If pi_ hase been reduced significantly over the iteration, then terminate if pi_<0.005
        if pi_ <= 0.005:
            break
        # Calculate the sub-gradients for each constraints
        subgrads = np.array([0 if lm[i] == 0 and 1 - sum(x_array_lm[c] for c in data['rows'][i]) < 0 else 1 - sum(x_array_lm[c] for c in data['rows'][i]) for i in data['rows_index']])  #Formula 1
    #     subgrads = np.array([1 - sum(x_array_lm[c] for c in rows[i]) for i in data['rows_index']]) #Formula 2
        verboseprint(f"Subgradients at iteration {ic}: {subgrads}")
        sg_ss = np.square(subgrads).sum() # Sum od Squares of the SUb-gradirents
        print(f"SG Square Sum for iteration {ic}: {sg_ss}")
        # If all Sug-grads are zero, terminate
        if sg_ss == 0:
            break
        T = pi_ * (1.05 * ub_ - llbp_obj) / sg_ss #Calculate the step-size
        print(f"step-size T at iteration {ic}:  {round(T,2)}")
        lm = np.array([max(0,lm[i] + T*subgrads[i]) for i in data['rows_index']]) #Update the Lagrange Multupliers
    #     print(f"New multipliers of iteration {i}: {lm}")
        end  = pd.Timestamp('now') - start #Cumulative Time so far
        print(f"Time after iteration {ic} is: {end.components.hours} hours {end.components.minutes} minutes {end.components.seconds} seconds")
        print("="*100 + "\n")
        ic+=1
    end  = pd.Timestamp('now') - start #Total time 
    print(f"Time taken for LR algorithm is: {end.components.hours} hours {end.components.minutes} minutes {end.components.seconds} seconds")
    print("^*^"*40)
    
    if plotResult == True:
        plt.clf()
        plt.figure(figsize=(18,8))
        plt.title("Best Lower Bounds over iterations")
        plt.plot(lbs[1:])
        plt.show()
        
        plt.clf()
        plt.figure(figsize=(18,8))
        plt.title("LLBP Lower Bounds over iterations")
        plt.plot(llbpObjs[1:])
        plt.show()
        
        plt.clf()
        plt.figure(figsize=(18,6))
        plt.title("Best Upper Bounds over iterations")
        plt.plot(ubs[1:])
        plt.show()
        
        plt.clf()
        plt.figure(figsize=(18,6))
        plt.title("Heuristic Upper Bounds over iterations")
        plt.plot(ubs[1:])
        plt.show()
    
    out = {}
    for j in data['items']:
        if best_solution[j] == 1:
            out[j] = 1
    return  z_ub, out
    
if __name__ == '__main__':
    data = get_data()
    # data = read_data("scpnre1.txt")
    print("SCP MIP pyomo: ")
    print(mip_pyomo(data,verbose=True))
    print("="*150)
    print()
    print("SCP MIP OR tools: ")
    print(mip_OR_Tools(data,verbose=True))
    print("="*150)
    print()
    print("LR Method: ")
    print(LR_method(data = data, lmi = data['t'],verbose=False,plotResult=True))
    print("="*150)
    print()
        