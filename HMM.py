# **************************************
# Huston Bokinsky
# CSCE771, Summer 2012
# Dr. Matthews
# 12 July, 2012
# Function and class definitions related to
#    Hidden Markov Models and their uses.
# **************************************

from __future__ import division
import math
import random

def make_chain(N):
    """Build state path and observation sequence for ice cream model.

    Uses 'true' probability values from p. 178 J&M.
    """
    # initialize list to represent days of summer
    summer = list()
    for d in range(N): summer.append(dict())

    # set state of first day from <start> -> day_1 transition
    p = random.random()
    if p < 0.8: summer[0]['state'] = 'H'
    else      : summer[0]['state'] = 'C'

    # calculate states of remaining days based on transition from
    # day before state
    for day in range(1, N):
        p = random.random()
        if summer[day - 1]['state'] == 'H':
            if p < 0.7: summer[day]['state'] = 'H'
            else: summer[day]['state'] = 'C'
        else:
            if p < 0.6: summer[day]['state'] = 'C'
            else: summer[day]['state'] = 'H'

    # generate ice creams eaten as probable function of day state
    for day in summer:
        p = random.random()
        if day['state'] == 'H':
            if p < 0.2: day['eis'] = 1
            elif p < 0.6: day['eis'] = 2
            else: day['eis'] = 3
        else:
            if p < 0.5: day['eis'] = 1
            elif p < 0.9: day['eis'] = 2
            else: day['eis'] = 3

    return summer

# To construct model from J&M p. 178:
# m = HMM(['H', 'C'], [1, 2, 3])
# t = {'H':{'H':0.7, 'C':0.4}, 'C':{'H':0.3, 'C':0.6}}
# m.set_T(t)
# e = {1:{'H':0.2, 'C':0.5}, 2:{'H':0.4, 'C':0.4}, 3:{'H':0.4, 'C':0.1}}
# m.set_E(e)
# p = {'H':0.8, 'C':0.2}
# m.set_priors(p)

# To test against instructor manual solution...:
# f = forward([3,3,1,1,2,2,3,1,3], m)
# f['H'][8] + f['C'][8]
# ''.join(viterbi_path([3,3,1,1,2,2,3,1,3], m))
# ''.join(viterbi_path([3,3,1,1,2,3,3,1,3], m))

class HMM:
    """Class to implement an HMM.
    Defined by:
        1. Hidden state transition probability matrix T
        2. Observable emission probability matrix E
        3. Prior probability matrix 'priors'
        4. Vocabulary of possible hidden states M ('states')
        5. Vocabulary of possible observable emissions V ('emissions')
    """
    def __init__(self, states, emissions):
        self._states = states
        self._emissions = emissions
        self._T = dict()
        self._E = dict()
        self._priors = dict()
        print "Don't forget to set: T, E, and priors..."

    def set_T(self, new_T):
        tf = True
        for key in new_T:
            if sorted(new_T[key].keys()) != sorted(self._states):
                tf = False
        if sorted(new_T.keys()) != sorted(self._states):
            tf = False

        if tf: self._T = new_T
        else:
            print """Unmatched key -- check dictionary!
            T => T[to state][given state]
            """

    def set_E(self, new_E):
        tf = True
        for key in new_E:
            if sorted(new_E[key].keys()) != sorted(self._states):
                tf = False
        if sorted(new_E.keys()) != sorted(self._emissions):
            tf = False

        if tf: self._E = new_E
        else:
            print """Unmatched key -- check dictionary!
            E => E[emission][given state]
            """

    def set_priors(self, new_priors):
        if sorted(new_priors.keys()) == sorted(self._states):
            self._priors = new_priors
        else:
            print """Unmatched key -- check dictionary!
            priors => priors[state]
            """

# **************************************************
# Functions that take an observation sequence and an HMM
# **************************************************

def forward(O, hmm):
    """Return trellis representing p(theta_t | O_1^t).
    """
    # initialize local variables
    n = len(O)
    f = {state: list() for state in hmm._states}
    for o in O:
        for state in hmm._states:
            f[state].append(0)

    # construct forward trellis
    for state in hmm._states:
        f[state][0] = hmm._priors[state] * hmm._E[O[0]][state]
    for t in range(1, n):
        for j in hmm._states:
            for i in hmm._states:
                f[j][t] += f[i][t-1] * hmm._T[j][i] * hmm._E[O[t]][j]
    return f

def backward(O, hmm):
    """Return trellis representing p(O_t+1^N | theta_t == i).
    """
    # initialize local variables
    n = len(O)
    b = {state: list() for state in hmm._states}
    for o in O:
        for state in hmm._states:
            b[state].append(0)

    # construct backward trellis
    for state in hmm._states:
        b[state][n-1] = 1
    for t in range(n-2, -1, -1):
        for i in hmm._states:
            for j in hmm._states:
                b[i][t] += b[j][t+1] * hmm._T[j][i] * hmm._E[O[t+1]][j]
    return b

def posterior(O, hmm):
    """Return trellis representing p(theta_t | O).

    Posterior probabilities would be used to find the maximum likelihood
        of a state at a given time step based on the observation sequence
        O.  The value returned by the forward algorithm is the O_prob
        value returned here.
    """
    # get n
    n = len(O)
    # initialize forward, backward, and posterior trellises
    f = forward(O, hmm)
    b = backward(O, hmm)
    p = {state: list() for state in hmm._states}
    for o in O:
        for state in hmm._states:
            p[state].append(0)

    # total probability of sequence O
    O_prob = math.fsum([f[state][n-1] for state in hmm._states])

    # build posterior trellis
    for state in hmm._states:
        for t in range(n):
            p[state][t] = (f[state][t] * b[state][t]) / O_prob

    return p

def viterbi_path(O, hmm):
    """Return most likely hidden state path given observation sequence O.
    """
    n = len(O)
    u = {state: list() for state in hmm._states}
    v = {state: list() for state in hmm._states}
    bt = list()
    for o in O:
        for state in hmm._states:
            for t in (u, v):
                u[state].append(0)
                v[state].append(str())
            bt.append(str())

    for state in hmm._states:
        u[state][0] = hmm._priors[state] * hmm._E[O[0]][state]
        # v[state][0] not of interest
    for t in range(1, n):
        for j in hmm._states:
            for i in hmm._states:
                p = u[i][t-1] * hmm._T[j][i] * hmm._E[O[t]][j]
                if p > u[j][t]:
                    u[j][t] = p
                    v[j][t] = i
    p = 0
    for state in hmm._states:
        if u[state][n-1] > p:
            p = u[state][n-1]
            bt[n-1] = state
    for t in range(n-2, -1, -1):
        bt[t] = v[bt[t+1]][t+1]

    return bt

def baum_welch(O, hmm):
    """Return new hmm from one iteration of re-estimation."""
    n = len(O)
    f = forward(O, hmm)
    b = backward(O, hmm)
    p = posterior(O, hmm)
    E_prime = dict()
    for emission in hmm._emissions: E_prime[emission] = dict()
    T_prime = dict()
    for state in hmm._states: T_prime[state] = dict()
    priors_prime = dict()

    # construct E_prime
    for state in hmm._states:
        den = math.fsum([p[state][t] for t in range(n)])
        for emission in hmm._emissions:
            v = 0
            for t in range(n):
                if O[t] == emission: v += p[state][t]
            E_prime[emission][state] = v / den

    # construct T_prime
    p_O = math.fsum([f[s][n-1] for s in hmm._states])
    for given in hmm._states:
        den = math.fsum([p[given][t] for t in range(n)])
        for to in hmm._states:
            v = 0
            for t in range(1, n):
                v += ( f[given][t-1] *
                       b[to][t] *
                       hmm._T[to][given] *
                       hmm._E[O[t]][to]
                      ) / p_O
            T_prime[to][given] = v / den

    # construct priors_prime
    for state in hmm._states:
        priors_prime[state] = p[state][0]

    new_hmm = HMM(hmm._states, hmm._emissions)
    new_hmm.set_E(E_prime)
    new_hmm.set_T(T_prime)
    new_hmm.set_priors(priors_prime)

    return new_hmm
