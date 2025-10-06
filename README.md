# RLFAP Constrait Solver

This repository contains a program that solves the ** Radio Link Frequency Assignment Problems** (**rlfap**). This is a constraint satisfaction problem,
and we use this type of modeling to solve it using algorithms such as backtracking, backtracking with CBJ etc... 
This project was a part of Artificial Intelidgence I Course in my university, the assignment can be found here https://cgi.di.uoa.gr/~ys02/askiseis2023/hw3-2023.pdf.

## Algorithms Presented

The file rlfap_solver.py can solve the RLFAP using the following algorithms:

- Backtracking
- Backtracking with heuristics such as forward checking (FC), MAC and
- Backgracking with FC and conflict directed backjumping (CBJ).
- A local search solution MinConflicts. This algorithms fails to find solutions at most cases, because
  rlfap is a complex problem with a sparce solution space.


## Comments for the code

- For variable selection in our backtracking based algorithms, we use dynamic variable ordeering, using the
  heuristic dom/wdeg https://frontiersinai.com/ecai/ecai2004/ecai04/pdf/p0146.pdf

- rlfap_solver.py uses a lot of code from https://github.com/aimacode/aima-python/blob/master/csp.py
 
- There are some metrics added in the code, to give us the ability to compare those algorithms in our
  assignment, such as the number of nodes visited, the number of assigns that happened and the number of
  constraint checks.

  
# Executing Code

The code was built for solving the rlfap problem using pre-given data sets, so if you want to use it with your own sets, 
you probably need to change the main file.
