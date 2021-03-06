# import the relevant packages
from scipy.stats import norm
import numpy as np

# Black-Scholes Model
def bsm(S0,K,T,r,q,sigma,opType):
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    c = (S0 * np.exp(-q*T) * norm.cdf(d1)) - (K*np.exp(-r*T)*norm.cdf(d2))
    if opType=='call':
        return c
    else: 
        p = c - (S0 * np.exp(-q*T)) + (K*np.exp(-r*T)) # put call parity with dividend yield
        return p

    
# Monte-Carlo Simulation
def MonteCarlo(S0,K,T,r,q,sigma,opType,N,M): # N is the number of simulations
    dt = T/M # time length of steps
    wiener = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(N,M))
    S = S0 * np.exp( np.sum( (r-q-0.5*sigma**2)*dt + sigma*wiener,axis=1 ) )
    if opType=='call':
        return np.exp(-r*T) * np.mean(np.maximum(S-K, 0))
    else:
        return np.exp(-r*T) * np.mean(np.maximum(K-S, 0))

# Binomial tree simulation
def binomial(S0,K,T,r,q,sigma,opType,M):
    dt = T/M # time length of steps
    u = np.exp((r-q)*dt + sigma*np.sqrt(dt))
    d = np.exp((r-q)*dt - sigma*np.sqrt(dt))
    p = (np.exp((r-q)*dt)-d)/(u-d)
    mu = np.arange((M+1))
    mu = np.resize(mu,(M+1,M+1)) #sets upper triangular matrix of number of up moves
    md = np.transpose(mu) #sets upper triangular matrix of number of down moves
    mu = u**(mu-md) #sets gross returns of up moves (upper triangular)
    md = d**md #sets gross returns o fdown moves (upper triangular)
    S = S0*mu*md #upper triangular matrix of asset prices (recombining)
    S = np.triu(S, k=0)
    payoff = np.zeros((M+1,M+1), dtype=object)
    arr = np.array([np.maximum(S[:,-1]-K, 0), np.maximum(K-S[:,-1], 0)]).T # terminal payoffs
    arr = list(arr) # convert to list of arrays of terminal call and put payoffs
    payoff[:,-1] = arr
    for column in range(M-1, -1, -1):
        for node in range(0, column+1):
            up = payoff[node, column+1]
            down = payoff[node+1, column+1]
            payoff[node,column] = np.exp(-r*dt) * ((p*up) + ((1-p)*down))
    if opType=='call':
        return payoff[0,0][0]
    else:
        return payoff[0,0][1]


# common input parameters
S0 = 103
K = 101
r = 0.05
q = 0.01
T = 1
sigma = 0.15
opType = 'call'

print('\nThe Black-Scholes ' + opType + ' Price is:', bsm(S0,K,T,r,q,sigma,opType))
print('\n')
print('The Monte-Carlo ' + opType + ' Price is:', MonteCarlo(S0,K,T,r,q,sigma,opType,N,M))
print('\n')
print('The binomial ' + opType + ' price is:', binomial(S0,K,T,r,q,sigma,opType,M))
