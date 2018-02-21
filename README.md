# MO-ASMO
This is a standalone version of MO-ASMO, a surrogate based multi-objective optimization algorithm.

Quick start: please run ZDT1/ZDT1\_MOASMO.py to start your first run.

Many test cases with the test function ZDT1.
1. ZDT1\_NSGA2.py: Optimize with NSGA2 algorithm, a traditional multi-objective optimization algorithm.
2. ZDT1\_MOASMO.py: Optimize with MO-ASMO.
3. ZDT1\_WNSGA2.py: Optimize with WNSGA2, NSGA2 with weighted crowding distance, which can constrain the search region better than your default parameters.
4. ZDT1\_WMOASMO.py: Optimize with WMO-ASMO, MO-ASMO with weighted crowding distance.

Other files in directory ZDT1:
ZDT1.py: the test function ZDT1.
ZDT1.txt: the parameter name, lower bound, upper bound.
ZDT1\_true.py: the true Pareto frontier of ZDT1.
ZDT1\dft.txt: default value of 2 objectives of ZDT1 function, used by WNSGA2 and WMO-ASMO. In WNSGA2 and WMO-ASMO, the search is constrained to the region that better than the objective values specified in this file.

Please cite this paper, if you use the code in your own research.
Gong, W., Q. Duan, J. Li, C. Wang, Z. Di, A. Ye, C. Miao, and Y. Dai (2016), Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, Water Resour. Res., 52(3), 1984-2008. doi:10.1002/2015WR018230.
