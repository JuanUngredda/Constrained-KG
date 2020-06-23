import numpy as np


def RMITD_core(x, runlength, seed):
    b = x[0]
    x = x[1:]
    noise = 0.3
    if (np.sum(x<0)>0) or runlength <=0 or runlength != int(runlength) or seed <= 0 or int(seed) != seed:
        print("x should be >= 0, runlength should be positive integer, seed must be a positive integer")
        fn = np.nan
        return fn
    else:
        nReps = runlength
        T = 3
        price = np.array([100, 300, 400])
        meanDemand = np.ones((nReps,3))*np.array([50, 20, 30])
        cost = 80
        buy = b
        x = np.ones((nReps,3))*x
        X = np.random.normal(0,noise,(nReps,3))
        revenue = np.zeros(nReps)
        remainingCapacity = np.ones((1,nReps))*buy
        for j in range(T):
            D_t = meanDemand[:,j] + X[:,j]
            D_t = D_t.reshape(-1)
            aux_vector = np.vstack((np.array(remainingCapacity - x[:,j]).reshape(-1), np.zeros(nReps)))
            max_val = np.max(aux_vector,axis=0).reshape(-1)
            aux_vector = np.vstack((max_val, D_t))
            sell = np.min(aux_vector, axis=0)
            remainingCapacity = np.array(remainingCapacity - sell).reshape(-1)
            revenue = revenue + price[j] * sell
        MeanRevenue = np.mean(revenue)
        fn = MeanRevenue - cost * buy
        return fn

# x = np.array([100,20,30,20])
# runlength = 10000
# seed=1
# print(RMITD(x, runlength, seed))


