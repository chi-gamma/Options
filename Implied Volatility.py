import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

# Funtion to calculate the Black-Scholes Price of an Option
def bsm(S0, K, T, r, sigma, opType):
    d1 = (np.log(S0/K) + ((r + 0.5*sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    c = (S0 * norm.cdf(d1,0,1)) - (K*np.exp(-r*T)*norm.cdf(d2,0,1))
    p = c - S0 + (K*np.exp(-r*T)) # put-call parity
    if opType == 'call':
        return c
    else:
        return p
    

# Function to determine the implied volatilitiy of an option price
def imp_vol(S0, K, T, r, prc, opType): # opType = 'call' or 'put'
    
    # Difference between Price using an estimate volatility and actual price
    def difference(sigma):
        prc_est = bsm(S0, K, T, r, sigma, opType)
        return prc_est - prc
    
    sigma_est = np.sqrt(2*np.pi/T) * (prc/S0) # a good initial guess of the IV
    
    iv = fsolve(difference, sigma_est)[0]
    return iv

# Test if the IV is close to the sigma used to calculate the price
S0, K, T, r, sigma, opType = 100, 97, 1, 0.05, 0.35, 'call'
prc = bsm(S0, K, T, r, sigma, opType)

eps = 10**-5
iv = imp_vol(S0, K, T, r, prc, opType)
if abs(iv - sigma) < eps:
    print('It Converged! The implied volatility of the option is: ', iv )
