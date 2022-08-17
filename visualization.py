import visdom
import numpy as np
import time

class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        # self.index = {}
        self.env = env
        self.log_text = ''

    def plot_one(self, x, y, name, xlabel='iter', ylabel=''):
        # x = self.index.get(name, 1)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 0 else 'append'
                      )
        # self.index[name] = x + step

    def plot_many_stack(self, x, d, xlabel='iter',  ylabel='',name_window=None):
        name = list(d.keys())
        name_total = " ".join(name)
        if name_window is None:
            name_window = name_total
        # x = self.index.get(name_total, 1)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y, X=np.ones(y.shape)*x,
                      win=str(name_window),
                      opts=dict(legend=name, title=name_window, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 0 else 'append'
                      )
        # self.index[name_total] = x+1

    def log(self, info, win='log_text'):
        self.log_text += ('[{}] {} <br>'.format(time.strftime('%m/%d_%H:%M:%S'),info))
        self.vis.text(self.log_text, win)

    def hist(self, x, win, title, numbins=30, xlabel='iter', ylabel=''):
        self.vis.histogram(
            X=x, 
            win=win,
            opts=dict(numbins=numbins,title=title,xlabel=xlabel,ylabel=ylabel)
            )