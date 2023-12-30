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
        
        self.weights = {} #Map that stores the weights
                         # for the dom/wdeg heuristic
        for c in conDict:
            self.weights[c] = 1 #Initializing weights

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
            if ((var,x) in self.conDict):
                self.weights[(var,x)] += 1
            else:
                self.weights[(x,var)] += 1



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
            if ((y,x) in csp.conDict):
                sum += csp.weights[(y,x)]
            else:
                sum += csp.weights[(x,y)]
    if (sum == 0):
        sum = 1
    return sum

def dom_wdeg_ordering(assignment,csp):
    minVal = +100000000
    min = first_unassigned_variable(assignment,csp) #!!!!!EXPLAIN
    for var in csp.variables:
        if var not in assignment:
            wd = wdeg(var,assignment,csp)
            dom = len(csp.neighbors[var])
            temp = dom / (2 * wd)
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
            if not csp.curr_domains[B]:
                csp.revise(var)
                return False
    return True


#Backtracking algorithm

import time

def backtracking_search(csp,select_unassigned_variable = mrv,order_domain_values = unordered_domain_values,inference = forward_checking):#no_inference):
    startTime = time.time()
    def backtrack(assignment,startTime):
        if len(assignment) == len(csp.variables):
            return assignment
        if (time.time() - startTime > 500):
            print("Didn't finish\n")
            return None
        var = select_unassigned_variable(assignment,csp)

        for value in order_domain_values(var,assignment,csp):
            if 0 == csp.nconflicts(var,value,assignment):
                csp.assign(var,value,assignment)
                removals = csp.suppose(var,value)
                if inference(csp,var,value,assignment,removals):
                    result = backtrack(assignment,startTime)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var,assignment)
        #Domain is empty
        # csp.revise(var)

        return None
    
    result = backtrack({},startTime)
    assert result is None or csp.goal_test(result)
    return result

def constraints(A,a,B,b,conDict):
    if ((A,B) in conDict):
        ctr = conDict[(A,B)]
    else:
        ctr = conDict[(B,A)]
    if (ctr[0] == "="):
        return abs(int(a) - int(b)) == int(ctr[1]) 
    else:
        return abs(int(a) - int(b)) > int(ctr[1]) 

import os

def my_main():
    print("we will use file 2-f24\n")
    variables = []
    #Creating the list variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir , "rlfap\\var2-f24.txt")
    var = open(file_path,"r")
    #str is every line

    file_path = os.path.join(script_dir , "rlfap\dom2-f24.txt")
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
    file_path = os.path.join(script_dir , "rlfap\ctr2-f24.txt")
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
        
    problem = CSP(variables,domains,neighbors,constraints,conDict)
    temp = backtracking_search(problem)
    print(temp)

    x = {}

    x = {'24': '30', '25': '268', '196': '58', '197': '296', '135': '86', '134': '324', '94': '282', '95': '44', '110': '310', '111': '72', '74': '324', '75': '86', '28': '268', '29': '30', '84': '72', '85': '310', '77': '142', '76': '380', '142': '268', '143': '30', '68': '156', '69': '394', '27': '100', '26': '338', '98': '142', '99': '380', '136': '30', '137': '268', '100': '156', '101': '394', '195': '128', '194': '366', '56': '142', '57': '380', '58': '254', '59': '16', '124': '114', '125': '352', '144': '100', '145': '338', '96': '30', '97': '268', '93': '114', '92': '352', '120': '128', '121': '366', '122': '30', '123': '268', '35': '254', '34': '16', '36': '86', '37': '324', '146': '30', '147': '268', '114': '30', '115': '268', '133': '100', '132': '338', '108': '114', '109': '352', '130': '30', '131': '268', '171': '100', '170': '338', '106': '16', '107': '254', '104': '114', '105': '352', '118': '30', '119': '268', '178': '72', '179': '310', '91': '282', '90': '44', '154': '338', '155': '100', '0': '16', '1': '254', '63': '352', '61': '254', '60': '16', '62': '114', '2': '128', '3': '366', '113': '268', '112': '30', '117': '296', '116': '58', '129': '324', '128': '86', '180': '128', '181': '366', '172': '268', '173': '30', '11': '100', '10': '338', '12': '254', '13': '16', '176': '44', '177': '282', '86': '128', '87': '366', '140': '100', '141': '338', '139': '30', '138': '268', '193': '30', '192': '268', '169': '100', '168': '338', '174': '30', '175': '268', '33': '72', '32': '310', '64': '254', '65': '16', '66': '156', '67': '394', '153': '114', '152': '352', '190': '44', '191': '282', '188': '114', '189': '352', '187': '30', '186': '268', '185': '100', '184': '338', '183': '30', '182': '268', '88': '100', '89': '338', '127': '114', '126': '352', '102': '30', '103': '268', '43': '58', '42': '296', '163': '142', '162': '380', '44': '296', '45': '58', '160': '296', '161': '58', '38': '310', '39': '72', '40': '310', '41': '72', '46': '394', '47': '156', '18': '16', '19': '254', '164': '86', '20': '16', '21': '254', '159': '156', '158': '394', '165': '324', '166': '16', '167': '254', '51': '282', '50': '44', '70': '268', '71': '30', '72': '268', '73': '30', '48': '30', '49': '268', '54': '142', '55': '380', '53': '16', '52': '254', '14': '44', '15': '282', '16': '142', '17': '380', '23': '58', '22': '296', '156': '16', '157': '254', '151': '16', '150': '254', '78': '30', '79': '268', '81': '100', '80': '338', '83': '114', '82': '352', '198': '30', '199': '268', '7': '16', '6': '254', '8': '86', '9': '324', '4': '30', '5': '268', '30': '44', '31': '282', '148': '30', '149': '268'}

    print(x == temp)
       
 

if __name__  == "__main__":
    my_main()