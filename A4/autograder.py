from bnetbase import *
from solution import *

test_multiply = True
test_restrict = True
test_sum = True
test_normalize = True
test_ve = True
test_nb = True
test_explore = True

#  E,B,S,w,G example
E, B, S, G, W = (Variable('E', ['e', '-e']), Variable('B', ['b', '-b']), Variable('S', ['s', '-s']),
                 Variable('G', ['g', '-g']), Variable('W', ['w', '-w']))
FE, FB, FS, FG, FW = (Factor('P(E)', [E]), Factor('P(B)', [B]), Factor('P(S|E,B)', [S, E, B]),
                      Factor('P(G|S)', [G, S]), Factor('P(W|S)', [W, S]))


FE.add_values([['e', 0.1], ['-e', 0.9]])
FB.add_values([['b', 0.1], ['-b', 0.9]])
FS.add_values([['s', 'e', 'b', .9], ['s', 'e', '-b', .2], ['s', '-e', 'b', .8], ['s', '-e', '-b', 0],
               ['-s', 'e', 'b', .1], ['-s', 'e', '-b', .8], ['-s', '-e', 'b', .2], ['-s', '-e', '-b', 1]])
FG.add_values([['g', 's', 0.5], ['g', '-s', 0], ['-g', 's', 0.5], ['-g', '-s', 1]])
FW.add_values([['w', 's', 0.8], ['w', '-s', .2], ['-w', 's', 0.2], ['-w', '-s', 0.8]])

SampleBN = BN('SampleBN', [E, B, S, G, W], [FE, FB, FS, FG, FW])


def test_multiply_B_E():
    print("\nMultiply Factors Test ... ", end='')
    factor = multiply_factors([FB, FE])
    tests = []
    values = []
    for e_val in E.domain():
        for b_val in B.domain():
            try:
                value = factor.get_value([e_val, b_val])
                values.append(value)
            except ValueError:
                value = factor.get_value([b_val, e_val])
                values.append(value)
        tests.append(value == FE.get_value([e_val]) * FB.get_value([b_val]))
    if all(tests):
        print("passed.")
    else:
        print("failed.")
    print('P(e,b) = {} P(-e,b) = {} P(e,-b) = {} P(-e,-b) = {}'.format(values[0], values[1], values[2], values[3]))


def test_sum_fun():
    print("\nSum Out Variable Test ....", end='')
    factor = sum_out_variable(FS, E)
    values = (factor.get_value(["s", "b"]), factor.get_value(["s", "-b"]), factor.get_value(["-s", "b"]),
              factor.get_value(["-s", "-b"]))
    tests = (abs(values[0] - 1.7) < 0.001, abs(values[1] - 0.2) < 0.001, abs(values[2] - 0.3) < 0.001,
             abs(values[3] - 1.8) < 0.001)
    if all(tests):
        print("passed.")
    else:
        print("failed.")
    print(
        'P(S = s | B = b) = {} P(S = s | B = -b) = {} P(S = -s | B = b) = {} '
        'P(S = -s | B = -b) = {}'.format(values[0], values[1], values[2], values[3]))


def test_restrict_FG_restrict_s():
    print("\nRestrict Factor Test ...", end='')
    factor = restrict_factor(FG, S, 's')
    value = factor.get_value_at_current_assignments()
    if value == 0.5:
        print("passed.")
    else:
        print("failed.")
    print('P(G|S=s) = {}'.format(value))


def test_normalize_fun():
    print("\nNormalize Test .... ", end='')
    normalized_nums = normalize([i for i in range(0, -5, -1)])
    norm_sum = sum(normalized_nums)
    if norm_sum == 1:
        print("passed.")
    else:
        print("failed.")
    print('{} when normalized to {} sum to {}'.format([i for i in range(0, -5, -1)], normalized_nums, norm_sum))


def test_ve_fun():
    print("\nVE Tests .... ")
    print("Test 1 ....", end='')
    S.set_evidence('-s')
    W.set_evidence('w')
    probs3 = VE(SampleBN, G, [S, W])
    S.set_evidence('-s')
    W.set_evidence('-w')
    probs4 = VE(SampleBN, G, [S, W])
    if probs3[0] == 0.0 and probs3[1] == 1.0 and probs4[0] == 0.0 and probs4[1] == 1.0:
        print("passed.")
    else:
        print("failed.")
    print('P(g|-s,w) = {} P(-g|-s,w) = {} P(g|-s,-w) = {} P(-g|-s,-w) = {}'.format(probs3[0], probs3[1], probs4[0],
                                                                                   probs4[1]))
    print("Test 2 ....", end='')
    W.set_evidence('w')
    probs1 = VE(SampleBN, G, [W])
    W.set_evidence('-w')
    probs2 = VE(SampleBN, G, [W])
    if abs(probs1[0] - 0.15265998457979954) < 0.0001 and abs(probs1[1] - 0.8473400154202004) < 0.0001 and abs(
            probs2[0] - 0.01336753983256819) < 0.0001 and abs(probs2[1] - 0.9866324601674318) < 0.0001:
        print("passed.")
    else:
        print("failed.")
    print('P(g|w) = {} P(-g|w) = {} P(g|-w) = {} P(-g|-w) = {}'.format(probs1[0], probs1[1], probs2[0], probs2[1]))


def test_ve_1():
    print("\nVE Tests .... ")
    nb = NaiveBayesModel()
    Salary = nb.get_variable("Salary")
    Work = nb.get_variable("Work")
    Education = nb.get_variable("Education")
    Relationship = nb.get_variable("Relationship")
    Occupation = nb.get_variable("Occupation")

    Work.set_evidence("Government")
    Education.set_evidence("<Gr12")
    Relationship.set_evidence("Husband")
    Occupation.set_evidence("Admin")
    probs = VE(nb, Salary, [Work, Education, Relationship, Occupation])
    print(probs[1])


def test_nb_fun():
    print("\nNaive Bayes Model Test .... ", end='')
    nb = NaiveBayesModel()
    prior_count = 0
    for f in nb.Factors:
        s = f.scope
        if len(s) == 1:
            prior_count += 1
        if len(s) > 2:
            print("failed.")
            return
    if prior_count > 1:
        print("failed.")
        return
    print("passed.")


def test_explore_fun():
    print("\nExplore Test .... \n")
    print(f"Q1: {Explore(NaiveBayesModel(), 1)}")
    print(f"Q2: {Explore(NaiveBayesModel(), 2)}")
    print(f"Q3: {Explore(NaiveBayesModel(), 3)}")
    print(f"Q4: {Explore(NaiveBayesModel(), 4)}")
    print(f"Q5: {Explore(NaiveBayesModel(), 5)}")
    print(f"Q6: {Explore(NaiveBayesModel(), 6)}")


if __name__ == '__main__':
    if test_multiply: test_multiply_B_E()
    if test_restrict: test_restrict_FG_restrict_s()
    if test_sum: test_sum_fun()
    if test_normalize: test_normalize_fun()
    if test_ve: test_ve_fun()
    if test_nb: test_nb_fun()
    if test_explore: test_explore_fun()
