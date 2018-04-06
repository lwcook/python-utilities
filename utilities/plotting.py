import matplotlib.pyplot as plt
import os
import datetime

blue = [136./255., 186./255., 235./255.]
orange = [253./255., 174./255., 97./255.]
red = [140./255., 20./255., 32./255.]
green = [171./255., 221./255., 164./255.]

def savefig(*args, **kwargs):
    return safeFig(*args, **kwargs)

def saveFig(name, fig_current=None, formatstr='pdf', plotsize=[3.3, 2.8]):
    if fig_current is None:
        fig_current = plt.gcf()

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if not os.path.isdir(os.getcwd() + "/figs/"):
        os.mkdir(os.getcwd() + "/figs/")
    name = os.getcwd() + "/figs/" + name.lstrip("/")
    name = name + '_' + str(date.day) + months[date.month-1] + '.' + formatstr

    fig_current.savefig(name, format=formatstr)

def saveaxis(*args, **kwargs):
    return saveAxis(*args, **kwargs)

def saveAxis(name, fig_current=None, ax_current=None,
        formatstr='pdf', expand=[1.1, 1.2], scale=True, plotsize=[3.3, 2.8]):
    if fig_current is None:
        fig_current = plt.gcf()
    if ax_current is None:
        ax_current = plt.gca()

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if not os.path.isdir(os.getcwd() + "/figs/"):
        os.mkdir(os.getcwd() + "/figs/")

    ## finish making auto re-scaling of axes to paper size when saving
    #if scale:
    #    fig_size = fig_current.size()
    #    orig_position = ax_current.get_position()
    #    ax_current.set_position(orig_position[0], orig_position[1], 3.3, 2.8)

    name = os.getcwd() + "/figs/" + name.lstrip("/")
    name = name + '_' + str(date.day) + months[date.month-1] + '.' + formatstr

    extent = ax_current.get_window_extent().transformed(fig_current.dpi_scale_trans.inverted())
    fig_current.savefig(name, bbox_inches=extent.expanded(*expand), format=formatstr)

    #ax_current.set_position(*orig_position)

def plotContours(f, lb, ub, N=50):
    xlin = np.linspace(lb[0], ub[0], N)
    ylin = np.linspace(lb[1], ub[1], N)

    xv, yv, zv = np.zeros([N, N]), np.zeros([N, N]), np.zeros([N, N])
    for ix, x in enumerate(xlin):
        for iy, y in enumerate(ylin):
            xv[iy, ix] = float(x)
            yv[iy, ix] = float(y)
            zv[iy, ix] = f([float(x), float(y)])

    return xv, yv, zv


def saveContours(xv, yv, zv, N, name='saved_contours',
                 bSaveBase=True, base='/phd-thesis/Figs/'):
    jdict = {'xv': [_ for _ in xv.flatten()],
             'yv': [_ for _ in yv.flatten()],
             'zv': [_ for _ in zv.flatten()],
             'N': N}
    savedata(jdict, name=name)


def readContours(name='saved_contours'):
    jdict = readdata(name=name)
    N = jdict['N']
    xv = np.array(jdict['xv']).reshape([N, N])
    yv = np.array(jdict['yv']).reshape([N, N])
    zv = np.array(jdict['zv']).reshape([N, N])
    return xv, yv, zv, N
