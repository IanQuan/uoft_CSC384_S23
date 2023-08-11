from bnetbase import Variable, Factor, BN
import csv

# DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
variable_domains = {
    "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Gender": ['Male', 'Female'],
    "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    "Salary": ['<50K', '>=50K']
}


def generate_assignments(scope):
    """Given the scope of a factor, which is a list of variables,
    generate all possible assignments to a list of variables."""

    if len(scope) == 0:
        return

    current_index = [0] * len(scope)  # keep track of the current index for each variable in scope
    init = False

    while True:
        if not init:  # no assignment have been checked before
            init = True
        else:
            for i in range(0, len(scope)):
                current_index[i] += 1
                if current_index[i] >= scope[i].domain_size():
                    if i == len(scope) - 1:
                        return
                    current_index[i] = 0
                else:
                    break
        assignment = [scope[i].dom[current_index[i]] for i in range(0, len(scope))]
        yield assignment


def multiply_two_factors(fac_1: Factor, fac_2: Factor):
    """Return a new factor that is the product of the fac_1 and fac_2.
    @return a factor"""
    new_scope = []
    scope_1 = fac_1.get_scope()
    scope_2 = fac_2.get_scope()

    # new_scope is now a set of variables in fac_1 + variables in fac_2
    for var in scope_1:
        new_scope.append(var)
    for var in scope_2:
        if var not in new_scope:
            new_scope.append(var)

    new_factor = Factor('[' + fac_1.name + ' x ' + fac_2.name + ']', new_scope)

    if not scope_1 and not scope_2:  # if both factors are constant factors, return the product of the values
        new_value = fac_1.values[0] * fac_2.values[0]
        new_factor.add_values([[new_value]])

    var_index_1 = [new_scope.index(var) for var in scope_1]  # index of the variable in scope_1 correspond to new_scope
    var_index_2 = [new_scope.index(var) for var in scope_2]  # index of the variable in scope_2 correspond to new_scope

    for assignment in generate_assignments(new_factor.scope):
        a1, a2 = [], []  # list of assignments in fac_1 and fac_2 respectively
        for i in var_index_1:
            a1.append(assignment[i])
        for j in var_index_2:
            a2.append(assignment[j])

        new_value = fac_1.get_value(a1) * fac_2.get_value(a2)
        if not scope_1:  # fac_1 is a constant factor
            new_value = fac_1.values[0] * fac_2.get_value(a2)
        elif not scope_2:  # fac_2 is a constant factor
            new_value = fac_1.get_value(a1) * fac_2.values[0]

        assignment.append(new_value)  # add the multiplied value to the end of the assignment list
        new_factor.add_values([assignment])

    # print('\n' + new_factor.name + ':')
    # new_factor.print_table()
    # print('---------------------------------------\n')

    return new_factor


def multiply_factors(Factors):
    """Factors is a list of factor objects.
    Return a new factor that is the product of the factors in Factors.
    @return a factor"""

    while Factors:
        if len(Factors) == 1:  # Base Case
            return Factors[0]
        else:  # Recursive Part
            fac_1 = Factors.pop()
            fac_2 = Factors.pop()

            # print("========= Multiply" + fac_1.name + " with " + fac_2.name + "========= ")
            # print('\n' + fac_1.name + ':')
            # fac_1.print_table()
            # print('---------------------------------------')
            # print('\n' + fac_2.name + ':')
            # fac_2.print_table()
            # print('---------------------------------------\n')

            Factors.append(multiply_two_factors(fac_1, fac_2))
    return None  # if Factors is empty


def restrict_factor(f, var, value):
    """f is a factor, var is a Variable, and value is a value from var.domain.
    Return a new factor that is the restriction of f by this var = value.
    Don't change f! If f has only one variable its restriction yields a
    constant factor.
    @return a factor"""

    new_scope = f.get_scope()  # get a copy of the scope
    new_name = '[' + f.name + ' restrict ' + var.name + '=' + value + ']'

    try:
        var_index = new_scope.index(var)  # get the index of the restricted variable in new_scope
    except ValueError:
        return f  # if the variable is not in the factor's scope -> return the given factor

    new_scope.pop(var_index)  # remove the restricted variable from new_scope
    new_factor = Factor(new_name, new_scope)

    if not new_scope:  # there is only one variable in f and the only variable has been removed
        new_factor.values[0] = f.get_value([value])
    else:
        for assignments in generate_assignments(new_scope):
            # copy assignment is used to get the value for the new factor
            # while assignments is the reduced assignments after restricting var=value
            copy_assignments = assignments.copy()  # copy should be missing the restricted assignment
            copy_assignments.insert(var_index, value)  # add the missing assignment back
            new_value = f.get_value(copy_assignments)

            assignments.append(new_value)
            new_factor.add_values([assignments])

    # print("========= Restrict ========= ")
    # print('\n' + f.name + ':')
    # f.print_table()
    # print('---------------------------------------\n\n' + new_factor.name + ':')
    # new_factor.print_table()
    # print('---------------------------------------\n')

    return new_factor


def sum_out_variable(f, var):
    """f is a factor, var is a Variable.
    Return a new factor that is the result of summing var out of f, by summing
    the function generated by the product over all values of var.
    @return a factor"""

    new_scope = f.get_scope()  # get a copy of the scope
    new_name = '[' + f.name + ' sum_out ' + var.name + ']'

    try:
        var_index = new_scope.index(var)  # get the index of the variable that will be summed out
    except ValueError:
        return f  # if the variable is not in the factor's scope -> return the given factor

    new_scope.pop(var_index)  # remove the variable that will be summed out
    new_factor = Factor(new_name, new_scope)

    for assignments in generate_assignments(new_scope):
        new_value = 0  # the new value of each assignment after summing out var
        for domain_value in var.domain():
            # copy assignment is used to find the value for the new factor
            # while assignments is reduced assignments after summing out var
            copy_assignments = assignments.copy()  # copy should be missing the assignment of the variable that will be summed out
            copy_assignments.insert(var_index, domain_value)  # add the missing assignment of the variable back
            new_value = f.get_value(copy_assignments) + new_value

        assignments.append(new_value)
        new_factor.add_values([assignments])

    # print("========= Sum Out ========= ")
    # print('\n' + f.name + ':')
    # f.print_table()
    # print('---------------------------------------\n\n' + new_factor.name + ':')
    # new_factor.print_table()
    # print('---------------------------------------\n')

    return new_factor


def normalize(nums):
    """num is a list of numbers. Return a new list of numbers where the new
    numbers sum to 1, i.e., normalize the input numbers.
    @return a normalized list of numbers"""
    if sum(nums) == 0:
        return [0] * len(nums)
    normalized_nums = [num / sum(nums) for num in nums]

    # print("========= Normalize ========= ")
    # print(normalized_nums)
    # print('\n')

    return normalized_nums


def get_hidden_vars(Factors, QueryVar):
    """Factors is a list of factor objects, QueryVar is a query variable.
    Variables in the list will be derived from the scopes of the factors in Factors.
    The QueryVar must NOT be part of the returned non_query_variables list.
    @return a list of variables"""

    scopes = []  # A list of list of variables across all the scopes in the factor of Factors
    non_query_variables = []  # A list of non-duplicated variables excluding QueryVar

    for factor in Factors:
        scopes.append(list(factor.get_scope()))

    # Get the list of non-query variables
    for scope in scopes:
        for var in scope:
            if not var in non_query_variables and var != QueryVar:
                non_query_variables.append(var)
    return non_query_variables


def VE(Net, QueryVar, EvidenceVars):
    """
    Input: Net---a BN object (a Bayes Net)
           QueryVar---a Variable object (the variable whose distribution
                      we want to compute)
           EvidenceVars---a LIST of Variable objects. Each of these
                          variables has had its evidence set to a particular
                          value from its domain using set_evidence.
     VE returns a distribution over the values of QueryVar, i.e., a list
     of numbers, one for every value in QueryVar's domain. These numbers
     sum to one, and the i'th number is the probability that QueryVar is
     equal to its i'th value given the setting of the evidence
     variables. For example if QueryVar = A with Dom[A] = ['a', 'b',
     'c'], EvidenceVars = [B, C], and we have previously called
     B.set_evidence(1) and C.set_evidence('c'), then VE would return a
     list of three numbers. E.g. [0.5, 0.24, 0.26]. These numbers would
     mean that Pr(A='a'|B=1, C='c') = 0.5 Pr(A='a'|B=1, C='c') = 0.24
     Pr(A='a'|B=1, C='c') = 0.26
     @return a list of probabilities, one for each item in the domain of the QueryVar
     """
    Factors = Net.factors().copy()

    # 1) Restrict the Factors over the evidence variables
    for i in range(len(Factors)):
        for evidence_var in EvidenceVars:
            scope = Factors[i].get_scope()
            if evidence_var in scope:  # check if the factor's scope contains the evidence variable
                evidence_value = evidence_var.get_evidence()  # get the assigned value of the evidence variable
                Factors[i] = restrict_factor(Factors[i], evidence_var, evidence_value)

    hidden_vars = get_hidden_vars(Factors, QueryVar)

    # 2) Eliminate variables
    for var in hidden_vars:
        factors_to_be_removed = []
        copy_factors = Factors.copy()

        # Search for the factors containing the hidden variable, and add them to factors_to_be_removed
        for i in range(len(Factors)):
            current_factor = copy_factors[i]
            if var in current_factor.get_scope():
                factors_to_be_removed.append(current_factor)
                Factors.remove(current_factor)  # update the list of Factors

        multiplied_factor = multiply_factors(
            factors_to_be_removed)  # multiply all the factors that contain hidden var to form a joint factor
        new_factor = sum_out_variable(multiplied_factor, var)  # sum out the factor over the var we are now eliminating
        Factors.append(new_factor)  # add the factor that have finished eliminating var back to Factors

    # remove factors that contains no variables since factors with no variables must be independent from the goal factor
    related_Factors = []
    for factor in Factors:
        if (len(factor.scope) != 0) or (len(factor.scope) == 0 and factor.values[0] != 0):
            related_Factors.append(factor)

    # 4) Normalize the probability of the final factor
    final_factor = multiply_factors(related_Factors)
    return normalize(final_factor.values)  # there should be only one factor remaining in final_factor


def create_variables(attributes, variable_domains):
    """Return a list of Variable objects with the name and domain instantiated"""

    variables = []  # list of Variable objects excluding Salary
    class_variable = None
    for attribute in attributes:
        new_var = Variable(attribute, variable_domains[attribute])
        if attribute == "Salary":  # use Salary as the prior
            class_variable = new_var
        else:
            variables.append(new_var)
    return variables, class_variable


def compute_conditional_prob(attribute_index, attribute_value, salary_value, dataset):
    """Given the salary_value and the attribute_value, return the conditional probability
    P(attribute=attribute_value | Salary=salary_value) by counting the frequency of the
    data that matches in dataset"""

    count_salary, count_attribute_and_salary = 0, 0

    for data in dataset:
        if data[-1] == salary_value:
            count_salary += 1
            if data[attribute_index] == attribute_value:
                count_attribute_and_salary += 1

    if count_salary == 0:
        return 0

    return count_attribute_and_salary / count_salary


def NaiveBayesModel():
    """
   NaiveBayesModel returns a BN that is a Naive Bayes model that 
   represents the joint distribution of value assignments to 
   variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
   assumes P(X1, X2,.... XN, Class) can be represented as 
   P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
   When you generated your Bayes Net, assume that the values 
   in the SALARY column of the dataset are the CLASS that we want to predict.
   @return a BN that is a Naive Bayes model and which represents the Adult Dataset. 
    """

    # Read in training data
    input_data = []
    with open('data/adult-train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    # In a naive Bayes network, each attribute is conditionally independent of all other
    # attributes given the class variable (Salary)

    # 1) Define one variable for each attribute
    # variables is a list of Variable objects excluding Salary
    variables, class_variable = create_variables(headers, variable_domains)

    # 2) Define one factor for each variable
    factors = []  # list of Factor objects excluding Salary
    for var in variables:
        factors.append(Factor('P(' + var.name + '|Salary)', [var, class_variable]))
    class_factor = Factor('P(Salary)', [class_variable])

    # 3) Populate the conditional probability table in each factor by counting the frequency of values in the training set
    count_salary_less_than_50K = 0
    count_salary_more_than_50K = 0
    total = len(input_data)
    for data in input_data:
        if data[-1] == '<50K':
            count_salary_less_than_50K += 1
        elif data[-1] == '>=50K':
            count_salary_more_than_50K += 1
    # Create the prior probability with Salary
    class_factor.add_values([['<50K', count_salary_less_than_50K / total],
                             ['>=50K', count_salary_more_than_50K / total]])

    # Create a dictionary the matches the attributes to their corresponding index in the list in input_data
    attribute_to_index = {}
    for index, attribute in enumerate(headers):
        attribute_to_index[attribute] = index

    # Compute conditional probability for each attribute given Salary
    for salary_value in ['<50K', '>=50K']:
        for factor in factors:  # iterate through all the attributes (variables)
            # Get the attribute name from the factor
            full_name = factor.name
            attribute = full_name[full_name.index("(") + 1:full_name.index("|Salary)")].strip()
            # Get the index of the attribute in the header list
            attribute_index = attribute_to_index[attribute]
            # print(f"\n P({attribute} = value | Salary = {salary_value}):")

            for attribute_value in variable_domains[attribute]:  # iterate through all the domain values of the attribute
                prob = compute_conditional_prob(attribute_index, attribute_value, salary_value, input_data)
                factor.add_values([[attribute_value, salary_value, prob]])
                # print(f"P({attribute} = {attribute_value} | Salary = {salary_value}): {prob:.4f}")

    return BN("Predict_Salary", variables + [class_variable], factors + [class_factor])


def get_assigned_vars(data, variables, index_dict, is_E2):
    """Given a list of Variable objects, assign the corresponding values
    based on the input data provided.
    E1: [Work, Occupation, Education, and Relationship Status]
    E2: [Work, Occupation, Education, Relationship Status, and Gender]"""

    assigned_vars = []  # A list of Variable object in E1 or E2 with value assigned
    var_salary = None
    # i = 0  # for testing
    for var in variables:
        if var.name in ["Work", "Occupation", "Education", "Relationship"]:
            var_index = index_dict[var.name]
            var.set_evidence(data[var_index])
            # var.set_evidence(data[i])  # for testing
            assigned_vars.append(var)
            # i += 1  # for testing
        if is_E2:
            if var.name == "Gender":
                var_index = index_dict["Gender"]
                var.set_evidence(data[var_index])
                assigned_vars.append(var)
        if var.name == "Salary":
            var_salary = var

    return assigned_vars, var_salary


def Explore(Net, question):
    """Input: Net---a BN object (a Bayes Net)
    question---an integer indicating the question in HW4 to be calculated. Options are:
    1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
    2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
    3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
    4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
    5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
    6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
    @return a percentage (between 0 and 100)
    """

    # Read in test data
    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)  # skip header row
        for row in reader:
            input_data.append(row)

    # Create a dictionary the matches the attributes to their corresponding index in the list in input_data
    attribute_to_index = {}
    for index, attribute in enumerate(headers):
        attribute_to_index[attribute] = index

    variables = Net.variables()  # from the bayes net model (include Salary)
    count_women_E1_greater_E2, count_men_E1_greater_E2 = 0, 0
    count_women_predict, count_men_predict = 0, 0
    count_women_actual, count_men_actual = 0, 0
    count_women_total, count_men_total = 0, 0
    result = 0
    index = 0

    # ---------------- For Testing: Get the E1 Distribution ---------------------
    # test_Q1, test_Q2,test_Q3, test_Q4, test_Q5, test_Q6 = [], [], [], [], [], []  # for testing

    # with open('E1_output.txt', 'w') as file:
    #     domain_values = []
    #     for var in variable_domains.keys():
    #         if var in ["Work", "Occupation", "Education", "Relationship"]:
    #             domain_values.append(variable_domains[var])
    #     permutations = list(itertools.product(*domain_values))
    #     for data in permutations:
    #         assigned_vars, var_salary = get_assigned_vars(data, variables, attribute_to_index, is_E2=False)
    #         prob_E1 = VE(Net, var_salary, assigned_vars)
    #         assigned_values = [var.get_evidence() for var in assigned_vars]
    #         file.write(f"{index}. {assigned_values}: {prob_E1[1]}\n")
    # ---------------------------------------------------------------------------

    with open('E1_output.txt', 'w') as file:
        file.write("E2 distribution\n")
        for data in input_data:
            # E1 prediction
            assigned_vars, var_salary = get_assigned_vars(data, variables, attribute_to_index, is_E2=False)
            prob_E1 = VE(Net, var_salary, assigned_vars)

            if prob_E1[1] > 0.5:  # P(Salary=">=$50K"|E1)
                if data[attribute_to_index["Gender"]] == "Female":
                    count_women_predict += 1
                    # test_Q5.append(index)  # for testing
                    if data[attribute_to_index["Salary"]] == ">=50K":
                        count_women_actual += 1
                        # test_Q3.append(index)  # for testing
                elif data[attribute_to_index["Gender"]] == "Male":
                    count_men_predict += 1
                    # test_Q6.append(index)  # for testing
                    if data[attribute_to_index["Salary"]] == ">=50K":
                        count_men_actual += 1
                        # test_Q4.append(index)  # for testing

            if question in [1, 2]:
                # E2 prediction
                assigned_vars, var_salary = get_assigned_vars(data, variables, attribute_to_index, is_E2=True)
                prob_E2 = VE(Net, var_salary, assigned_vars)

                if prob_E1[1] > prob_E2[1]:  # P(Salary=">=$50K"|E1) > P(Salary=">=$50K"|E2)
                    if data[attribute_to_index["Gender"]] == "Female":
                        count_women_E1_greater_E2 += 1
                        # test_Q1.append(index)  # for testing
                    elif data[attribute_to_index["Gender"]] == "Male":
                        count_men_E1_greater_E2 += 1
                        # test_Q2.append(index)  # for testing

            if data[-3] == "Female":
                count_women_total += 1
            elif data[-3] == "Male":
                count_men_total += 1
            index += 1

    if question == 1:
        result = count_women_E1_greater_E2 / count_women_total
    elif question == 2:
        result = count_men_E1_greater_E2 / count_men_total
    elif question == 3:
        result = count_women_actual / count_women_predict
    elif question == 4:
        result = count_men_actual / count_men_predict
    elif question == 5:
        result = count_women_predict / count_women_total
    elif question == 6:
        result = count_men_predict / count_men_total

    # with open('Explore_result.txt', 'w') as file:
    #     file.write(f'Q1 = {count_women_E1_greater_E2 / count_women_total * 100}, {test_Q1}\n'
    #                f'Q2 = {count_men_E1_greater_E2 / count_men_total * 100}, {test_Q2}\n'
    #                f'Q3 = {count_women_actual / count_women_predict * 100}, {test_Q3}\n'
    #                f'Q4 = {count_men_actual / count_men_predict * 100}, {test_Q4}\n'
    #                f'Q5 = {count_women_predict / count_women_total * 100}, {test_Q5}\n'
    #                f'Q6 = {count_men_predict / count_men_total * 100}, {test_Q6}\n')

    return result * 100
