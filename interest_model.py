
import numpy as np

from explauto.utils import rand_bounds
from explauto.interest_model.interest_model import InterestModel
from explauto.interest_model.random import RandomInterest
from explauto.interest_model.competences import competence_dist
from explauto.models.dataset import BufferedDataset as Dataset


class MiscRandomInterest(RandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest in the whole space,
    the recent competence on the babbled points in the whole space, 
    the competence around a given point based on a mean of the knns.   
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 competence_measure,
                 win_size,
                 competence_mode,
                 k,
                 progress_mode):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.competence_measure = competence_measure
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.k = k
        self.progress_mode = progress_mode
        self.data_xc = Dataset(len(expl_dims), 1)
        self.data_sr = Dataset(len(expl_dims), 0)
        self.current_progress = 0.
        self.current_interest = 0.
              
            
    def add_xc(self, x, c):
        self.data_xc.add_xy(x, [c])
        
    def add_sr(self, x):
        self.data_sr.add_xy(x)
        
    def update_interest(self, i):
        self.current_progress += (1. / self.win_size) * (i - self.current_progress)
        self.current_interest = abs(self.current_progress)

    def update(self, xy, ms, snnp=None, sp=None):
        c = self.competence_measure(xy[self.expl_dims], ms[self.expl_dims], dist_max=self.dist_max)
        if self.progress_mode == 'local':
            interest = self.interest_xc(xy[self.expl_dims], c)
            self.update_interest(interest)
        elif self.progress_mode == 'global':
            pass
        
        self.add_xc(xy[self.expl_dims], c)
        self.add_sr(ms[self.expl_dims])
        return interest
    
    def n_points(self):
        return len(self.data_xc)
    
    def competence_global(self, mode='sw'):
        if self.n_points() > 0:
            if mode == 'all':
                return np.mean(self.data_c)
            elif mode == 'sw':
                idxs = range(self.n_points())[- self.win_size:]
                return np.mean([self.data_xc.get_y(idx) for idx in idxs])
            else:
                raise NotImplementedError
        else:
            return 0.
        
    def mean_competence_pt(self, x):
        if self.n_points() > self.k: 
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            return np.mean([self.data_xc.get_y(idx) for idx in idxs])
        else:
            return self.competence()
                
    def interest_xc(self, x, c):
        if self.n_points() > 0:
            idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
            sr_NN = self.data_sr.get_x(idx_sg_NN)
            c_old = competence_dist(x, sr_NN, dist_max=self.dist_max)
            return c - c_old
            #return np.abs(c - c_old)
        else:
            return 0.
        
    def interest_pt(self, x):
        if self.n_points() > self.k:
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            idxs = sorted(idxs)
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        else:
            return self.interest_global()
            
    def interest_global(self): 
        if self.n_points() < 2:
            return 0.
        else:
            idxs = range(self.n_points())[- self.win_size:]
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        
    def competence(self): return self.competence_global()
        
    def interest(self):
        if self.progress_mode == 'local':
            return self.current_interest
        elif self.progress_mode == 'global':
            return self.interest_global()
        else:
            raise NotImplementedError
        
        
