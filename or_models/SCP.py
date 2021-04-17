import numpy as np
import time
import math
from ortools.linear_solver import pywraplp

def verboseprint_func(verbose):
    """
    Set verobsity.

    Args:
        verbose (bool): Verbosity {True, False}

    Returns:
        function: verboseprint function that prints as per verbosity.
    """
    if verbose:
        def verboseprint(*args, **kwargs):
            # print the args and kwargs
            print(*args, **kwargs)
    else:
        def verboseprint(*args, **kwargs):
            # do-nothing function
            verboseprint = lambda *a, **k: None
    return verboseprint



class SCP:
    """
    Set Covering Problem Model Object.

    Implementing Set Covering Problem using various techniques:
        - MIP Model using Google OR-tools.
        - Using Lagrange Relaxation and Lagrangean Heuristic to get a good feasible solution.
    
    References:
        - Integer programming solution methods - J E Beasley - http://people.brunel.ac.uk/~mastjjb/jeb/natcor_ip_rest.pdf)
        - Beasley, John E. "A lagrangian heuristic for set‐covering problems." Naval Research Logistics (NRL) 37.1 (1990): 151-164. 
        - Lagrangian relaxation can solve your optimization problem much, much faster - https://medium.com/bcggamma/lagrangian-relaxation-can-solve-your-optimization-problem-much-much-faster-daa9edc47cc9)
        - Geoffrion, Arthur M. "Lagrangian relaxation for integer programming." 50 Years of Integer Programming 1958-2008. Springer, Berlin, Heidelberg, 2010. 243-281.
        - Fisher, Marshall L. "The Lagrangian relaxation method for solving integer programming problems." Management science 27.1 (1981): 1-18.
        - Beasley datasets: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html    
    """

    def __init__(self,costs,matrix):
        """
        Create a SCP model object.

        Args:
            costs (np.array): Array of costs of length 'n'
            matrix (np.array): A matrix array of dimension 'm x n' with values as either 0 or 1

        Returns:
            None: Creates a SCP object
        """
        self.probName = 'SCP'
        self.verboseprint = verboseprint_func(False)

        self.costs = np.copy(costs)
        self.matrix = np.copy(matrix)

        self.n = len(costs)
        self.m = matrix.shape[0]

        self.matrix_transpose = np.transpose(self.matrix)

        self.rows_covered = np.empty((self.n,),dtype='O')
        for j in range(self.n):
            self.rows_covered[j] = np.where(self.matrix_transpose[j] == 1)[0]

        self.cols_covering = np.empty((self.m,),dtype='O')
        for i in range(self.m):
            self.cols_covering[i] = np.where(self.matrix[i] == 1)[0]

        self.row_least_cost = np.zeros((self.m,), dtype=np.float64)
        self.row_least_cost_var =np.zeros((self.m,), dtype=int)
        
        for i in range(self.m):
            temp = self.costs * self.matrix[i]
            temp_min = np.amin(temp[np.nonzero(temp)])
            self.row_least_cost[i] = temp_min
            self.row_least_cost_var[i] = np.where(temp == temp_min)[0][0]
        
        
        # % gap between the best lower bound and best upper bound at each iteration
        self.gaps = []

        # Lower Bounds of each iteration are stored here
        self.llbpObjs = []

        # Upper Bounds of each iteration are stored here 
        self.lhObjs = []

        # Best Lower Bounds of each iteration are stored here
        self.lbs = []

        # Best Lower Bounds of each iteration are stored here
        self.ubs = []

        # Google OR outputs
        self.GOR_objValue = None
        self.GOR_output = None

        # LR method outputs
        self.LR_objValue = None
        self.LR_output = None
        return None

    def set_verbose(self, verbose=False):
        """
        Set verbose print argument for the model outputs.

        Args:
            verbose (bool, optional): Argument that sets the verbosity level. Defaults to False.
        
        Examples:
            >>> model.set_verbose(True)
        """
        self.verboseprint = verboseprint_func(verbose)
    
    @classmethod
    def read_data(cls,file_loc):
        """
        Construct a SCP model from Beaslet dataset file.

        Args:
            file_loc (str): Location to dataset file.
            Beasley datasets: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html    

        Returns:
            or_models.SCP: A SCP model object created for the file_loc dataset

        Examples:
            >>> scp_model = SCP.read_data("scpnre1.txt")
        """
        file_location = fr"{file_loc}".strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        lines = input_data.split('\n')
        
        # Read first line
        count = 0
        m, n = lines[count].split()
        m, n = int(m), int(n)
        count += 1
        costs = []
        while len(costs) < n:
            parts = lines[count].split()
            count += 1
            costs += parts
        costs = np.array(list(map(float, costs)))
        
        rows = []
        while len(rows) < m:
            cols = int(lines[count].split()[0])
            count+=1
            i = 0
            row = []
            while len(row) < cols:
                row += lines[count].split()
                count+=1
            rows.append(row)
        
        for i in range(len(rows)):
            rows[i] = np.array(list(map(int, rows[i])))
            rows[i] = np.array(list(map(lambda x: x - 1, rows[i])))
        
        matrix = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                if j in rows[i]:
                    matrix[i][j] = 1

        return cls(costs,matrix)
    
    def solve_with_GOR(self,
                       time_limit = 1800,
                       logs = False):
        """
        Solves the SCP using Google OR tools.

        Returns:
            (GOR_objValue,GOR_output): Objective value and list of variables selected to be 1
        """                
        #Declare Model
        solver = pywraplp.Solver(self.probName,pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) 
        
        #Enable Log Printing (May not work)
        if logs:
            solver.EnableOutput()

        # Set Time Limit for the Solver Run
        solver.set_time_limit(time_limit*1000) 
        
        # Variable Declaration
        x = {}
        for j in range(self.n):
            x[j] = solver.IntVar(0,1,'x[%i]' %j) 
        
        # Constraints Declaration
        constraints = {} 
        for i in range(self.m):
            constraint_expr = [self.matrix[i][j] * x[j] for j in range(self.n)]
            constraints[i] = solver.Add(sum(constraint_expr) >= 1)
        
        # Objective Function Declaration
        solver.Minimize(solver.Sum(self.costs[j] * x[j] for j in range(self.n)))
        
        # OR-tools solver status dict
        dict_status = {0:"Optimal", 1:"Feasible"}  
        
        # Solve the model
        status = solver.Solve()
        
        # If the output is Optimal or Feasible
        if status == pywraplp.Solver.FEASIBLE or status == pywraplp.Solver.OPTIMAL:
            print("Status: ",dict_status[status])
            print('Objective value =', solver.Objective().Value())
            self.verboseprint('Problem solved in %f milliseconds' % solver.wall_time())
            self.verboseprint('Problem solved in %d iterations' % solver.iterations())
            self.verboseprint('Problem solved in %d branch-and-bound nodes' % solver.nodes())
            print()
        
        # In case it is not feasible or optimal
        else:
            print('The problem does not have an optimal solution.')
            print()
        
        # Get the objective and variables values
        try:
            self.GOR_objValue = solver.Objective().Value()
            self.GOR_output = []
            for j in range(self.n):
                if x[j].solution_value() == 1:
                    self.GOR_output.append(1)
                else:
                    self.GOR_output.append(0)
        
        # In case the model exited abnormally
        except:
            self.GOR_objValue = None
            self.GOR_output = None
            print("Something went wrong! Google OR tools didn't exit with a valid solution.")
            print()
        
        self.verboseprint(f"Variables Values: {self.GOR_output}")
        
        return self.GOR_objValue, self.GOR_output
    

    def _solve_llbp(self,costs,lm):
        C = np.array([costs[j] - sum(lm[i] for i in self.rows_covered[j]) for j in range(self.n)])
        self.verboseprint("LLBP costs: ",C)
        self.verboseprint("Lagrange Multipliers: ",lm)
        x_array_llbp = (C<=0).astype(int)
        lb = np.nan_to_num(x_array_llbp*C).sum() + np.nan_to_num(lm).sum()
        self.verboseprint("LLBP Onjective: ", lb)
        self.verboseprint("LLBP solution: ",x_array_llbp)
        return lb , x_array_llbp, C

    def _heuristic(self,costs,x_array_lm):
        x_local = np.copy(x_array_lm)
        for i in range(self.m):
            if sum(x_local[j] for j in self.cols_covering[i]) == 0:
                x_local[self.row_least_cost_var[i]] = 1
        x_ones = -np.sort(-np.array(np.where(x_local==1)[0]))
        self.verboseprint("Indices of variables that are 1 in LLBP Solution: ",x_ones)
        for j in x_ones:
            isImp = False
            for i in self.rows_covered[j]:
                if sum(x_local[k] for k in self.cols_covering[i]) - x_local[j] == 0:
                    isImp = True
                    break
            if not isImp:
                x_local[j] = 0
        ub = np.nan_to_num(costs * x_local).sum()
        self.verboseprint("Bound from the heuristic: ",ub)
        self.verboseprint("Variable Values from the Heuristic: ",x_local)
        return ub , x_local

    def LR_method(self,lmi,numIterations = 1000):
        """
        Run the Lagrange Relaxation method on the SCP instance.

        Args:
            lmi (np.array): 1-D array of length 'm' with initial values of Lagrange Multipliers.
            numIterations (int, optional): The number of iterations for the algortihm. Defaults to 1000.

        Returns:
            (float,np.array): The objective value from the LR method and values of the decision variables.

        Examples:
        >>> costs = np.array([2,3,4,5])
        >>> matrix = np.array([[1,0,1,0],
                               [1,0,0,1],
                               [0,1,1,1]])
        >>> scp = SCP(costs,matrix)
        >>> output = scp.LR_method(scp.row_least_cost)
        >>> print(output)
        (5, {0: 1, 1: 1})

        """
        start  = time.process_time() # Save the time-stamp

        # Copy the initial Lagrange-Multipliers
        lm =np.copy(lmi)
        
        self.verboseprint(f"Initial multipliers: {lm}")
        self.verboseprint("*"*100)
        
        # Intitial value of pi which is a scaling factor for the step-size
        pi_ = 2

        # A placeholder for Loop Upper Bounds as +Infinity
        ub_ = math.inf 

        # Set the initial Upper Bound as -Infinity
        z_ub = math.inf  
        
        # Set the initial Lower Bound as -Infinity
        z_lb = -math.inf

        # Variable Penalties as their costs initially
        P = np.copy(self.costs)
        costs = np.copy(self.costs)

        # Iteration Counter
        ic = 0



        # Compare best LBs of the last "steps" iterations every iteration. If there is no improvement , then set: pi_ := pi_/2
        steps = 30

        while ic<=numIterations:

            self.verboseprint(f"Iteration {ic}")
            self.verboseprint("-"*100)

            #Solve LLBP - Lagrange Lower Bound Problem
            llbp_obj, x_array_lm, C = self._solve_llbp(costs,lm) 
            
            # Update the best lower bound
            z_lb = max(llbp_obj,z_lb) 
            self.llbpObjs.append(llbp_obj)

            # Store the best lower bound for the iteration
            self.lbs.append(z_lb) 
            
            if ic % 10 == 0 or self.lbs[-1] > self.lbs[-2]:
                """
                If the iteration count is a multiple of 10 OR we find an improvement in the lower bound, then
                Run the Lagrange Heuristic to generate a feasible Solution from the LLBP solution.
                """
                # Run the Lagrangean Heuristic
                ub_ , x_array_h_= self._heuristic(costs,x_array_lm)

                # Update the best upper bound
                if ub_ < z_ub:
                    z_ub = ub_
                    best_solution = x_array_h_
            
            self.lhObjs.append(ub_)

            self.ubs.append(z_ub) # Store the best upper bound for the iteration
            
            self.verboseprint(f"LR heuristic objective at iteration {ic}: {round(ub_,2)}")
            self.verboseprint(f"Upper Bound at iteration {ic}: {round(z_ub,2)}")
            self.verboseprint()
            self.verboseprint(f"LLBP Solution at iteration {ic}: {round(llbp_obj,2)}")
            self.verboseprint(f"Lower bound at iteration {ic}: {round(z_lb,2)}")

            # Calculate the gap between best UB and best LB.
            gap = (z_ub - z_lb) * 100 / z_ub 

            # Store the gap
            self.gaps.append(gap)

            self.verboseprint("Gap: ",gap) 
            if gap <= 2:
                # Exit the Method if gap is less than 2%
                break

            """
            Update the penalties of each variable.
            If the variable is selected in LLBP solution, Penalty = max(LLBP Objective, Curr_Penalty)
            If the variables is not selected in LLBP solution, Penalty = max(LLBP Objective + LLBP Co-efficient, Curr_Penalty)
            """
            
            P = np.array([max(P[j], llbp_obj + C[j]) if x_array_lm[j] == 0 else max(P[j], llbp_obj) for j in range(self.n)])
            self.verboseprint(f"Penalty (P) for iteration {ic}: {P}")

            """
            Update the costs of variables.
            If the Penalty of a variable is more than the best known upper bound,
            then the variable can be removed by setting its cost as infinity
            """

            costs = np.array([np.inf if P[j] > z_ub else costs[j] for j in range(self.n)])
            self.verboseprint(f"Costs after iteration {ic}: {costs}")

            # If No improvement in best Lower Bound last "steps" iteration, set pi_ := pi_/2
            if ic > steps:
                if all(ele == self.lbs[-30] for ele in self.lbs[-30:]):
                    steps = ic+30
                    pi_ = pi_/2
            self.verboseprint(f"pi_ for iteration {ic}: {pi_}")

            # If pi_ hase been reduced significantly over the iteration, then terminate if pi_<0.005
            if pi_ <= 0.005:
                break
            
            # Calculate the sub-gradients for each constraints
            # Formula 1
            subgrads = np.array([0 if lm[i] == 0 and 1 - sum(x_array_lm[j] for j in self.cols_covering[i]) < 0 else 1 - sum(x_array_lm[j] for j in self.cols_covering[i]) for i in range(self.m)])  
            
            # Formula 2
            # subgrads = np.array([1 - sum(x_array_lm[c] for c in rows[i]) for i in data['rows_index']]) #Formula 2
            self.verboseprint(f"Subgradients at iteration {ic}: {subgrads}")

            # Sum od Squares of the SUb-gradirents
            sg_ss = np.square(subgrads).sum() 
            self.verboseprint(f"SG Square Sum for iteration {ic}: {sg_ss}")
            # If all Sug-grads are zero, terminate
            if sg_ss == 0:
                break
            T = pi_ * (1.05 * ub_ - llbp_obj) / sg_ss #Calculate the step-size
            self.verboseprint(f"step-size T at iteration {ic}:  {round(T,2)}")
            lm = np.array([max(0,lm[i] + T*subgrads[i]) for i in range(self.m)]) #Update the Lagrange Multupliers
            # print(f"New multipliers of iteration {i}: {lm}")
            end  = time.process_time() - start #Cumulative Time so far
            self.verboseprint(f"Time after iteration {ic} is: {end//3600} hours {(end%3600)//60} minutes {(end%3600)%60} seconds")
            self.verboseprint("="*100 + "\n")
            ic+=1
        end  = time.process_time() - start #Total time 
        print(f"Time taken for LR algorithm is: {end//3600} hours {(end%3600)//60} minutes {(end%3600)%60} seconds")
        print("^*^"*40)
        
        self.LR_objValue = z_ub
        self.LR_output = {}
        for j in range(self.n):
            if best_solution[j] == 1:
                self.LR_output[j] = 1
        return  self.LR_objValue, self.LR_output
    


    



    



