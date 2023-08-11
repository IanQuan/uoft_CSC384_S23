from BayesianNetwork import *

##Implement all of the following functions

## Do not modify any of the objects passed in as parameters!
## Create a new Factor object to return when performing factor operations

SUMMED_COUNTER = 0

'''
multiply_factors(factors)

Parameters :
              factors : a list of factors to multiply
Return:
              a new factor that is the product of the factors in "factors"
'''
def multiply_factors(factors):
    while factors:
        if len(factors) == 1:
            return factors[0]
        else:
            # multiply the last two factors in the list and append the results.
            factors.append(compose_factors(factors.pop(), factors.pop()))

    # case: input factors is an empty list
    return None

'''
compose_factors(factor1, factor2) 

Parameters :
              factor1: the first factor
              factor2: the second factor
Return:
              a new factor that is the product of the two factors
'''
def compose_factors(factor1, factor2):
    newScope = []
    newName = "tempFactor"

    scope1 = factor1.get_scope()
    scope2 = factor2.get_scope()
    # get the new scope of the new factor
    for var1 in scope1:
        newScope.append(var1)
    for var2 in scope2:
        if not var2 in newScope:
            # add the variables that factor1 has but factor2 doesn't
            newScope.append(var2)

    # the order of the new scope: [var in f1 + var in f2 but not in f1]
    newFactor = Factor(newName, newScope)

    # find the corresponding position in newScope of vars in scope1 and scope2
    index_nf_s1 = helper_find_cor_position(scope1, newScope)
    index_nf_s2 = helper_find_cor_position(scope2, newScope)

    # for each assignment of the new factor, find the corresponding assignments
    #   of f1 and f2, and calculate the new value of the new assignment
    for assignment in newFactor.get_assignment_iterator():
        # find the corresponding assignment in f1 and f2
        a1 = []
        a2 = []
        for i in index_nf_s1:
            a1.append(assignment[i])
        for j in index_nf_s2:
            a2.append(assignment[j])

        newVal = factor1.get_value(a1) * factor2.get_value(a2)
        newFactor.add_value_at_assignment(newVal, assignment)

    return newFactor

'''
helper function for compose_factors() to find the corresponding new_scope positions of variables in old_scope. 
The result list of indices will be used to extract assignment that belongs to original factor from the assignments of new factor

Parameters :
              old_scope: the old_scope that is a part of new_scope
              new_scope: the scope that need to be searched
Return:
              a list of index of new_scope where the variables of old_scope locaed in the new_scope, ordered as the variables in old_scope.
'''
def helper_find_cor_position(old_scope, new_scope):
    return [new_scope.index(var) for var in old_scope]


'''
restrict_factor(factor, variable, value):

Parameters :
              factor : the factor to restrict
              variable : the variable to restrict "factor" on
              value : the value to restrict to
Return:
              A new factor that is the restriction of "factor" by
              "variable"="value"
      
              If "factor" has only one variable its restriction yields a 
              constant factor
'''
def restrict_factor(factor, variable, value):
    newScope = factor.get_scope()
    newName = factor.name + "_res_" + variable.name + "_" + str(value)

    # find the position of the original variable and replace the new variable with it.
    varIndex = newScope.index(variable)
    newScope.pop(varIndex)
    newFactor = Factor(newName, newScope)

    if not newScope:    # case: only one variable in the scope and it has been popped
        newFactor.values[0] = factor.get_value([value])
    else:               # case: multiple variables in the scope
        for assignment in newFactor.get_assignment_iterator():
            old_a = list(assignment)
            old_a.insert(varIndex, value)
            newVal = factor.get_value(old_a)
            newFactor.add_value_at_assignment(newVal, assignment)

    return newFactor

'''
helper function to sum the probabilities up given the factor and the variable and its value

Parameters :
              factor : the factor to sum up
              variable : the variable to sum up
              value : the value of the variable to sum up
Return:
              the sum of the probabilities over the factor given the variable has the value
'''

def helper_sum_given_value(factor, variable, value):
    scope = factor.get_scope()

    # find the position of the original variable and replace the new variable with it.
    varIndex = scope.index(variable)
    scope.pop(varIndex)
    tempFactor = Factor(factor.name, scope)

    # calculate the sum of the probabilities while the variable equals the value
    sum_pr_given_val = 0
    for assignment in tempFactor.get_assignment_iterator():
        old_a = list(assignment)
        # form an old assignment of factor with the given value on the variable
        old_a.insert(varIndex, value)
        sum_pr_given_val += factor.get_value(old_a)

    return sum_pr_given_val

'''    
sum_out_variable(factor, variable)

Parameters :
              factor : the factor to sum out "variable" on
              variable : the variable to sum out
Return:
              A new factor that is "factor" summed out over "variable"
'''
def sum_out_variable(factor, variable):
    global SUMMED_COUNTER
    SUMMED_COUNTER += 1
    newName = "fp_" + str(SUMMED_COUNTER)

    # the new scope of the new factore should not contain the variable
    newScope = factor.get_scope()
    # remember the index of the variable
    varIndex = newScope.index(variable)
    newScope.pop(varIndex)
    newFactor = Factor(newName, newScope)

    for assignment in newFactor.get_assignment_iterator():
        newVal = 0
        for domVal in variable.domain():
            old_a = list(assignment)
            # Form an old assignment that exists in the factor
            old_a.insert(varIndex, domVal)
            newVal = newVal + factor.get_value(old_a)
        newFactor.add_value_at_assignment(newVal, assignment)

    return newFactor


    
'''
VariableElimination(net, queryVar, evidenceVars)

 Parameters :
              net: a BayesianNetwork object
              queryVar: a Variable object
                        (the variable whose distribution we want to compute)
              evidenceVars: a list of Variable objects.
                            Each of these variables should have evidence set
                            to a particular value from its domain using
                            the set_evidence function. 

 Return:
         A distribution over the values of QueryVar
 Format:  A list of numbers, one for each value in QueryVar's Domain
         -The distribution should be normalized.
         -The i'th number is the probability that QueryVar is equal to its
          i'th value given the setting of the evidence
 Example:

 QueryVar = A with Dom[A] = ['a', 'b', 'c'], EvidenceVars = [B, C]
 prior function calls: B.set_evidence(1) and C.set_evidence('c')

 VE returns:  a list of three numbers. E.g. [0.5, 0.24, 0.26]

 These numbers would mean that Pr(A='a'|B=1, C='c') = 0.5
                               Pr(A='b'|B=1, C='c') = 0.24
                               Pr(A='c'|B=1, C='c') = 0.26
'''       
def VariableElimination(net, queryVar, evidenceVars):
    factors = net.factors()

    # 1. Restriction of the factors over evidence variables
    for i in range(0, len(factors)):
        for var in evidenceVars:
            if var in factors[i].get_scope():
                value = var.get_evidence()
                factors[i] = restrict_factor(factors[i], var, value)

    # 2. Calculate the fixed ordering of the variables that want to be eliminated
    orderedVars = min_fill_ordering(factors, queryVar)

    # 3. Eliminate variables according to the ordering
    for var in orderedVars:
        # a temp list to store factors will be removed
        rmFactors = []
        factorsCopy = list(factors)

        # 3.1 Find the factors containing the variable
        for i in range(0, len(factors)):
            # iterate over the copy of the factors
            curFac = factorsCopy[i]
            scope = curFac.get_scope()
            if var in scope:
                rmFactors.append(curFac)
                factors.remove(curFac)

        # 3.2.1. Form the new factor with the factors containing the variable
        toSumFac = multiply_factors(rmFactors)

        # 3.2.2. Sum out the factor over the var we are now eliminating
        newFactor = sum_out_variable(toSumFac, var)

        # 3.3. Append this newFactor to Factors
        factors.append(newFactor)

    # 4. Normalize the answer
    finalFactor = multiply_factors(factors)

    totalSum = 0
    for assignment in finalFactor.get_assignment_iterator():
        totalSum += finalFactor.get_value(assignment)

    return [finalFactor.get_value([value]) * 1/totalSum for value in queryVar.domain()]
