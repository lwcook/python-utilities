import numpy as np
import time

def custom_target(x, q0=0, q1=1, p=2):

    if x < 0:
        return q0
    elif x > 1:
        return q1
    else:
        if p > 0:
            return q0 + (q1 - q0)*(x)**p
        else:
            return q1 + (q0 - q1)*(1-x)**abs(p)

def mean_estimator(vqsamp):
    return np.mean(vqsamp)


def var_estimator(vqsamp):
    return np.var(vqsamp)


def hm_estimator(vqsamp, targ=None):

    vq = np.sort(vqsamp, kind='mergesort')

    if targ is not None:
        ft = lambda h: targ(h)
    else:
        ft = lambda h: custom_target(h, 0, 0, 5)

    M = len(vqsamp)
    vh = [(1./M)*(0.5 + j) for j in range(M)]
    vt = np.array([float(ft(hi)) for hi in vh])
    Dhat = sum([(1./M)*(vq[j] - vt[j])**2 for j in range(len(vq))])

    Dhat = np.sqrt(Dhat)

    return float(Dhat)


def quantile_estimator(vqsamp, quantile=0.9):

    M = len(vqsamp)
    vq = np.sort(vqsamp, kind='mergesort')
    vh = [(1./M)*(j+1) for j in range(M)]
    for ii, (q, h) in enumerate(zip(vq, vh)):
        if h > quantile:
            Dhat = vq[ii]
            break

    return float(Dhat)


def bootstrap(f, vs, num, seed=None):
    """Bootstrapping on the function f with the given samples v_uj
        - f: function to be evaluated
        - vs: list of given samples of underlying uncertainty
        - num: number of times to bootstrap """

    resamples = samplebootstrap(vs, num, seed)
    boot = evalbootstrap(f, resamples, num)

    return boot, resamples


def samplebootstrap(vs, num, seed=None):

    if seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    M = len(vs)
    resamples = []
    for i in range(num):
        vsboot = []
        vrandj = np.floor(np.random.random(M)*M)
        for j, sj in enumerate(vs):
            randj = int(vrandj[j])
            vsboot.append(vs[randj])
        resamples.append(vsboot)

    return resamples


def evalbootstrap(f, mresamples, num):

    boot = np.zeros(num)
    for i in range(num):
        boot[i] = f(mresamples[i])

    return boot
