#This file gives a base solution of the problem
# using only backtracking

import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

class CSP:
    def __init__(self,variables,domains,neighbors,constraints,conDict):
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.conDict = conDict
        self.curr_domains = None
        self.nassigns = 0        
        #For the dom/wdeg heuristic
        self.weights = {} #Map that stores the weights
                         # for the dom/wdeg heuristic
        for c in conDict:
            self.weights[c] = 1 #Initializing weights
        
        self.depth = {} #To track each variables depth (we use dynamic ordering)
        #For the cbj mechanism
        self.conflicts = {}
        for x in variables:
            self.conflicts[x] = set()

    def assign(self,var,val,assignment):
        assignment[(var)] = val
        self.nassigns += 1
    
    def unassign(self,var,assignment):
        if var in assignment:
            del assignment[var]
    
    def nconflicts(self,var,val,assignment):
        def conflict(var2):
            return var2 in assignment and not self.constraints(var,val,var2,assignment[var2],self.conDict)
        
        count = 0
        for v in self.neighbors[var]:
            if conflict(v):
                count += 1
        return count
    
    def display(self,assignment): #For debuging probably
        print(assignment)

    
    def actions(self, state):
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = None
            for v in self.variables:
                if v not in assignment:
                    var = v
        return [(var,val) for val in self.domains[var]
                if self.nconflicts(var,val,assignment) == 0]
        
    def result(self, state, action):
        (var,val) = action
        return state + ((var, val),)

    def goal_test(self,state):
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # Constraint propagation

    def support_pruning(self):
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

   

    def suppose(self,var,value):
        self.support_pruning()
        removals = [(var,a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals
        
    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]


    # def choices(self, var):
    #     """Return all values for var that aren't currently ruled out."""
    #     if (self.curr_domains == None):
    #         if (var in self.domains):
    #             return (self.domains)[var]
    #         return []
    #     return (self.curr_domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    def revise(self,var):
        for x in self.neighbors[var]:
            self.weights[(var,x)] += 1
            self.weights[(x,var)] += 1

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]



#Functions for backtracking search
    
#For simple backtracking (We will use dom/wdeg heuristic instead)

def first_unassigned_variable(assignment, csp): 
    for var in csp.variables:
        if var not in assignment:
            return var
        
#Implementing dom/Wdeg heuristic


def wdeg(x,assignment,csp):
    sum = 0
    for y in csp.neighbors[x]:
        if (y not in assignment):
            sum += csp.weights[(y,x)]
    if (sum == 0):
        sum = 1
    return sum

def dom_wdeg_ordering(assignment,csp):
    minVal = +100000000
    min = first_unassigned_variable(assignment,csp) #!!!!!EXPLAIN
    for var in csp.variables:
        if var not in assignment:
            wd = wdeg(var,assignment,csp)
            dom = len(csp.domains[var])
            temp = dom / wd
            if (minVal > temp):
                minVal = temp
                min = var
    return min



from random import randint

def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    min = None
    l1 = []
    for v in csp.variables:
        if v not in assignment:
            if (min == None):
                min = v
                l1 = [v]
            if (num_legal_values(csp,min,assignment) > num_legal_values(csp,v,assignment)):
                min = v
                l1 = [v]
            elif (num_legal_values(csp,min,assignment) == num_legal_values(csp,v,assignment)):
                l1.append(v)
    # return l1[randint(0,len(l1) - 1)]
    return min

def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        count = 0
        for val in csp.domains[var]:
            if csp.nconflicts(var,val,assignment) == 0:
                count += 1
        return count

#Value ordering

#We will use something else probably
def unordered_domain_values(var,assigment,csp):
    x = csp.choices(var)
    return csp.choices(var)
        
        


# Inference 

def no_inference(csp,var,value,assignment,removals):
    return True

def forward_checking(csp,var,value,assignment,removals):
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var,value,B,b,csp.conDict):
                    csp.prune(B,b,removals)
                    csp.conflicts[var].add(B)
            if not csp.curr_domains[B]:
                csp.revise(var)
                return False
    return True


#Backtracking algorithm

import time

def backtracking_search(csp,select_unassigned_variable = dom_wdeg_ordering,order_domain_values = unordered_domain_values,inference = forward_checking):#no_inference):
    startTime = time.time()
    def backtrack(assignment,startTime,depth):
        if len(assignment) == len(csp.variables):
            return (assignment,0)
        if (time.time() - startTime > 500):
            print("Didn't finish\n")
            return None
        var = select_unassigned_variable(assignment,csp)
        csp.depth[var] = depth
        for value in order_domain_values(var,assignment,csp):
            if 0 == csp.nconflicts(var,value,assignment):
                csp.assign(var,value,assignment)
                removals = csp.suppose(var,value)
                if inference(csp,var,value,assignment,removals):
                    result = backtrack(assignment,startTime,depth + 1)
                    if (result[1] > 0):
                        del csp.depth[var]
                        csp.unassign(var,assignment)
                        return (None,result[1] - 1)
                    if result[0] is not None:
                        return (result[0],0)
                csp.restore(removals)
            # else:
            #     for b in assignment:
            #         if not csp.constraints(var,value,b,assignment[b]):
            #             if b not in csp.conflicts[var]:
            #                 csp.conflicts[var].append(b)

        del csp.depth[var]
        maxDepth = 0
        max = None
        for b in csp.conflicts[var]:

            if (b in csp.depth and maxDepth < csp.depth[b]):
                maxDepth = csp.depth[b] 
                max = b
        if max != None:
            csp.conflicts[max] = csp.conflicts[max].union(csp.conflicts[var]) 
        csp.unassign(var,assignment)
        #Domain is empty
        # csp.revise(var)   

        return (None,0)
    import datetime
    result,depth = backtrack({},startTime,0)
    convert = str(datetime.timedelta(seconds = time.time() - startTime))
    print("Running time: ",convert)
    assert result is None or csp.goal_test(result)
    return result

def constraints(A,a,B,b,conDict):
    ctr = conDict[(A,B)]
    if (ctr[0] == "="):
        return abs(int(a) - int(b)) == int(ctr[1]) 
    else:
        return abs(int(a) - int(b)) > int(ctr[1]) 

import os

def my_main():
    print("we will use file 3-f10\n")
    variables = []
    #Creating the list variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir , "rlfap\\var3-f10.txt")
    var = open(file_path,"r")
    #str is every line

    file_path = os.path.join(script_dir , "rlfap\dom3-f10.txt")
    dom = open(file_path,"r")
    #
    #Create the domains
    #(We will fix in forward check the domain-uniqueness)
    tempDomains = {} #A map that stores map[domain] = [domainValues]
    for subDom in dom:
        l1 = subDom.split()
        key = l1[0]
        l1.remove(key)
        tempDomains[key] = l1
    domains = {}
    first = False
    for str in var:
        if first == True:
            # temp = str.split()
            l1 = str.split()
            #The variables are a tuple of domain
            variables.append(l1[0])
            domains[l1[0]] = tempDomains[l1[1]]
        else:
            count = int(str)
            first = True
    
    #Create the neighbors and constraints function 
    file_path = os.path.join(script_dir , "rlfap\ctr3-f10.txt")
    ctr = open(file_path,"r")
    neighbors = {}
    conDict = {}
    for cons in ctr:
        l1 = cons.split()
        if (len(l1) > 1):
            temp = l1[0]
            if (temp in neighbors):
                neighbors[temp].append(l1[1])
            else:
                neighbors[temp] = [l1[1]]
            temp = l1[1]
            if (temp in neighbors):
                neighbors[temp].append(l1[0])
            else:
                neighbors[temp] = [l1[0]]
            l2 = l1
            conDict[(l1[0],l1[1])] = (l1[2],l1[3])
            conDict[(l1[1],l1[0])] = (l1[2],l1[3])
        
    problem = CSP(variables,domains,neighbors,constraints,conDict)
    print(backtracking_search(problem))
    # print(min_conflicts(problem))
 


def min_conflicts(csp,max_steps = 100000):
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp,var,current)
        prev = var
        csp.assign(var,val,current)
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        if (i % 100 == 0):
            print(i,") ",len(conflicted))
        var = random.choice(conflicted)
        val = min_conflicts_value(csp,var,current)
        csp.assign(var,val,current)
    return None

def min_conflicts_value(csp,var,current):
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))



def argmin_random_tie(seq, key):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)

def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items





    # min = []
    # # minVal = csp.nconflicts(var,val,current)
    # minVal = 10000000000
    # for val in csp.domains[var]:
    #     temp = csp.nconflicts(var,val,current)
    #     if (minVal > temp):
    #         min = [val]
    #         minVal = temp
    #     elif (minVal == temp):
    #         min.append(val)
    # return random.choice(min)

if __name__  == "__main__":
    my_main()

