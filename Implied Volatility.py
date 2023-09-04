import numpy as np
from scipy.stats import norm
from scipy.special import ndtr


def black_scholes_model(S, K, T, r, q, sigma, phi, vega=False):
    # phi = 1 for call options and -1 for put options
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.maximum(0.005, np.sqrt(T)))
    d2 = d1 - (sigma * np.maximum(0.005, np.sqrt(T)))
    val = phi * ( S * np.exp(-q * T) * ndtr(phi * d1) - (K * np.exp(-r * T) * ndtr(phi * d2)) )
    if vega:
        vg = S * norm._pdf(d1) * np.sqrt(T)
        return val, vg
    return val
 

def imp_vol_guess(opt_prc, S, K, r, q, T, phi, method='corrado-miller'):
    # phi = 1 for call options and -1 for put options
    phi = np.asarray(phi)
    X = K * np.exp(-r * T)
    C = np.copy(opt_prc)
    # Convert any put price to call price using the Put-Call parity formula
    if phi.shape==() and phi==-1: C = opt_prc + S * np.exp(-q * T) - X
    if phi.shape!=(): C[phi==-1] = opt_prc[phi==-1] + S * np.exp(-q * T) - X[phi==-1] 
    delta = (S - X) / 2
    if method=='corrado-miller':
        term1 = np.sqrt(2 * np.pi / T) / (S + X)
        term2 = C - delta
        term3_a = (C - delta)**2
        term3_b = (4 * delta**2) / np.pi
        sigma_est = term1 * (term2 + np.sqrt(np.maximum(1e-6, term3_a - term3_b))) 
    elif method=='brenner_subrahmanyam':
        sigma_est = np.sqrt(2 * np.pi / T) * ((C - delta) / S)
    sigma_est[(np.isnan(sigma_est)) | (sigma_est < 1e-5)] = 1e-5
    return sigma_est

    
def implied_volatility(opt_prcs, S, Ks, T, r, q, phis):
    # phi = 1 for call options and -1 for put options
    data = np.column_stack((opt_prcs, Ks, phis))
    inds = np.arange(0, len(data), dtype=np.int32)
    ivs = np.zeros(len(phis))
    sigmas_est = imp_vol_guess(opt_prcs, S, Ks, r, q, T, phis) # start guess
    max_iter = 100
    for i in range(max_iter):
        prc_est, vega = black_scholes_model(S, data[:,1], T, r, q, sigmas_est, data[:,2], vega=True)
        diff = data[:,0] - prc_est
        goal_flag = abs(diff) < 1e-5
        if np.any(goal_flag):
            # drop contracts that have already converged
            found_inds = inds[goal_flag]
            ivs[found_inds] = sigmas_est[goal_flag]
            inds = inds[~goal_flag]
            data = data[~goal_flag]
            sigmas_est = sigmas_est[~goal_flag] + (diff[~goal_flag] / vega[~goal_flag])
        else:
            sigmas_est = sigmas_est + (diff / vega)
    if len(sigmas_est)!=0: ivs[inds] = sigmas_est
    ivs[(np.isnan(ivs)) | (ivs < 1e-5)] = 1e-5
    return ivs


n = 100000      # number of option contracts
S0 = 100        # current price of asset 
r = 0.0536      # risk-free rate
q = 0.0154      # dividend yield
T = 1           # Time to maturity of contracts


# Generate random strikes, volatilities and option types(calls or puts)
Ks = S0 * np.random.uniform(0.5, 1.5, n)
sigmas = np.random.uniform(0.1, 0.55, n)
phis = np.random.choice([1,-1], size=n)


# Calculate the Black-Scholes price of the randomly generated option contracts
opt_prcs = black_scholes_model(S0, Ks, T, r, q, sigmas, phis)


# Calculate the Corrado-Miller guess of the implied volatility
iv_guess_cm = imp_vol_guess(opt_prcs, S0, Ks, r, q, T, phis)


# Calculate the Brenner-Subrahmanyam guess of the implied volatility
iv_guess_bs = imp_vol_guess(opt_prcs, S0, Ks, r, q, T, phis, method='brenner_subrahmanyam')


# starting with the Corrado-Miller or Brenner-Subrahmanyam initial guess calculate the 
# implied volatilities using a fast vectorized Newton-Raphson root finding algorithm
ivs = implied_volatility(opt_prcs, S0, Ks, T, r, q, phis)


# Calculate the Root-Mean-Square-Error
rmse_bs = np.sqrt(np.mean((sigmas - iv_guess_bs)**2))
rmse_cm = np.sqrt(np.mean((sigmas - iv_guess_cm)**2))
rmse_ivs = np.sqrt(np.mean((sigmas - ivs)**2))


print(f'\nThe Brenner-Subrahmanyam guess RMSE is {rmse_bs:.6f}')
print(f'\nThe Corrado-Miller guess RMSE is {rmse_cm:.6f}')
print(f'\nThe Implied Volatility RMSE is {rmse_ivs:.6f}')
