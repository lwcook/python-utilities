import datetime
import numpy as np

def finite_diff(*args, **kwargs):
    return finiteDiff(*args, **kwargs)

def finiteDiff(fobj, dv, f0=None, dvi=None, eps=10**-6):

    try:
        iter(dv)
    except:
        dv = [dv]

    if f0 is None: f0 = fobj(dv)
    if dvi is None:
        grad = []
        for ii in range(len(dv)):
            fbase = copy.copy(f0)
            x = copy.copy(dv)
            x[ii] += eps
            fnew = fobj(x)
            grad.append(float((fnew - fbase)/eps))
        if len(grad) == 1:
            return float(grad[0])
        else:
            return grad
    else:
        x = copy.copy(dv)
        x[dvi] += eps
        return (fobj(x) - f0) / eps


def minsmooth(a, b, eps=0.0000):
    return 0.5*(a + b - np.sqrt((a-b)**2 + eps**2))


def maxsmooth(a, b, eps=0.0000):
    return 0.5*(a + b + np.sqrt((a-b)**2 + eps**2))


def extalg(xarr, alpha=10):
    '''Given an array xarr of values, smoothly return the max/min'''
    return sum(xarr * np.exp(alpha*xarr))/sum(np.exp(alpha*xarr))


def extgrad(xarr, alpha=10):
    '''Given an array xarr of values, return the gradient of the smooth min/max
    swith respect to each entry in the array'''
    term1 = np.exp(alpha*xarr)/sum(np.exp(alpha*xarr))
    term2 = 1 + alpha*(xarr - extalg(xarr, alpha))

    return term1*term2


def rosenbrock(x):
    return (x[1]-x[0]**2)**2 + (1-x[0])**2


def lf_rosenbrock(x):
    return x[0]**4 + x[1]**2


def choose(n, k):
    if 0 <= k <= n:
        ntok, ktok = 1, 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def LHS(sample_num, dims=1):
    # Basic LHS that samples over the hypercube [-1,1]

    sample_points = np.zeros([sample_num, dims])
    permutations = np.zeros([sample_num, dims], int)

    # Using a uniform distribution
    for idim in range(dims):

        segment_size = 2. / float(sample_num)

        for isample in range(0, sample_num):

            segment_min = -1 + isample*segment_size
            sample_points[isample, idim] = segment_min + \
                np.random.uniform(0, segment_size)

        permutations[:, idim] = np.random.permutation(sample_num)

    temp = sample_points*0
    for isample in range(0, sample_num):
        for idim in range(0, dims):
            temp[isample, idim] = \
                sample_points[permutations[isample, idim], idim]
    sample_points = temp

    return sample_points

def getDateStr():

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    datestr = '_' + str(date.day) + months[date.month-1]

    return datestr
