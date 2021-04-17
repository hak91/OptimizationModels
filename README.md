# or_models (Operations Research Models)
Mathematical Optimization Models and Algorithms




## How to Install

You can use pip to install the code:
```
> pip install or_models
```

## Models Included:

1. [Set Covering Problem:](https://en.wikipedia.org/wiki/Set\_cover\_problem)
    
    Create an SCP object
    ```
    >>> costs = np.array([2,3,4,5])
    >>> matrix = np.array([[1,0,1,0],
                           [1,0,0,1],
                           [0,1,1,1]])
    >>> scp = SCP(costs,matrix)
    ```

    Solve with a MIP model created using Google OR-tools
    ```
    >>> scp.solve_with_GOR()
    ```
    Solve using Lagrange Relaxation method with some modification as discussed in
    
    *Beasley, John E. "A lagrangian heuristic for set‐covering problems." Naval Research Logistics (NRL) 37.1 (1990): 151-164.*
    ```
    >>> output = scp.LR_method(scp.row_least_cost,numIterations=1000)
    >>> print(output)
    (5, {0: 1, 1: 1})
    ```

## Dependencies

- google or-tools
- numpy
- time

