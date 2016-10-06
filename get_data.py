import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def get_data(X_path,drop=True):
    """Load data, mangle, add features."""

    # Column indexes for data set
    touch_index=['touch1','touch2','touch3','touch4','touch5','touch6','touch7']
    touch_att_index=['down_time','down_x','down_y','down_size','down_pressure',
      'up_time','up_x','up_y','up_size','up_pressure']
    touch_arrays=[touch_index,touch_att_index]
    non_touch_arrays=[['non_touch'],['user','screen_height','screen_width','timestamp']]
    tuples=list(product(*non_touch_arrays))+list(product(*touch_arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

    # load the data set
    X = pd.read_csv(X_path)
    X.columns=index

    #drop uninformative features
    X=X.drop(('non_touch','timestamp'),axis=1)
    X=X.drop(('non_touch','screen_height'),axis=1)
    X=X.drop(('non_touch','screen_width'),axis=1)
    #X_new = map_user_y(X,X['non_touch','user'][0])


    #time_diff features (start with first diff, move to others)
    for touch in touch_index:
        X[touch,'down_time']-=X['touch1','down_time']
        X[touch,'up_time']-=X['touch1','down_time']

    #angle features
    for touch in touch_index:
        X[touch,'down_angle']=np.arctan(X[touch,'down_y']/X[touch,'down_x'])
        X[touch,'up_angle']=np.arctan(X[touch,'up_y']/X[touch,'up_x'])

    #Relative Time between touch
    for (touch_prev,touch_post) in zip(touch_index[0:-1],touch_index[1:]):
        X[touch_post,'time_between_touch']=X[touch_post,'down_time']-X[touch_prev,'up_time']


    #Tme for a single touchs
    for touch in touch_index:
        X[touch,'down_up_time']=(X[touch,'up_time']-X[touch,'down_time'])


    # Average Size
    for touch in touch_index:
        X[touch,'area']=(X[touch,'up_size']+X[touch,'down_size'])

    # Feature Averages and variance
    X['touch_ave','time_between_touch']=np.average([X[x,'time_between_touch'] for x in touch_index[1:]],axis=0)
    X['touch_var','time_between_touch']=np.var([X[x,'time_between_touch'] for x in touch_index[1:]],axis=0)
    X['touch_ave','down_pressure']=np.average([X[x,'down_pressure'] for x in touch_index],axis=0)
    X['touch_ave','up_pressure']=np.average([X[x,'up_pressure'] for x in touch_index],axis=0)
    X['touch_var','down_pressure']=np.var([X[x,'down_pressure'] for x in touch_index],axis=0)
    X['touch_var','up_pressure']=np.var([X[x,'up_pressure'] for x in touch_index],axis=0)
    X['touch_ave','area']=np.average([X[x,'area'] for x in touch_index],axis=0)
    X['touch_var','area']=np.var([X[x,'area'] for x in touch_index],axis=0)
    X['touch_ave','up_size']=np.average([X[x,'up_size'] for x in touch_index],axis=0)
    X['touch_ave','down_size']=np.average([X[x,'down_size'] for x in touch_index],axis=0)


    #Combinations of Aggregate Features
    X['touch_ave','combo_pressure_area_var']=X['touch_var','down_pressure'] *X['touch_ave','area']
    X['touch_ave','combo_pressure_area']=X['touch_ave','down_pressure'] *X['touch_ave','area']
    X['touch_ave','combo_pressure_area_varu']=X['touch_var','down_pressure'] *X['touch_ave','up_size']
    X['touch_ave','combo_pressure_areau']=X['touch_ave','down_pressure'] *X['touch_ave','up_size']
    X['touch_ave','combo_pressure_area_vard']=X['touch_var','down_pressure'] *X['touch_ave','down_size']
    X['touch_ave','combo_pressure_aread']=X['touch_ave','down_pressure'] *X['touch_ave','down_size']
    X['touch_ave','combo_pressure_area_var-t']=X['touch_var','down_pressure'] *X['touch_var','time_between_touch']
    X['touch_ave','combo_pressure_area-t']=X['touch_ave','down_pressure'] *X['touch_ave','time_between_touch']
    X['touch_ave','down_pressure*tbt']=X['touch_ave','down_pressure'] *X['touch_var','time_between_touch']

    # Finally Drop all Non-Aggregate Features
    if drop:
        for x in touch_index:
            X=X.drop(x,level=0,axis=1)
            #energy features
    return X

# Data Augmentation, not used

# Map user strings to labels 1 or 0 in X
def map_user_y(X,name):
    X_new=X.copy()
    X_new['non_touch','user']=[int(x) for x in X_new['non_touch','user']==name]
    return X_new


# def
def visualise():
    X=get_data("dataset_training.csv",drop=False)
    touch_index=['touch1','touch2','touch3','touch4','touch5','touch6','touch7']
    users=X['non_touch','user'].unique()
    X=X[X['non_touch','user']==users[0]]
    i=1
    data_size=len(X)
    for touch in touch_index[1:]:
        plt.scatter([i for j in xrange(0,data_size)],X[touch,'time_between_touch'],)
        i+=1
    plt.pause(10)

visualise()
'''
def perm_touches(X):
    data_perms=[]

    for i in xrange(0,2000):
        ind_perm =np.random.permutation(touch_index)
        new_index=pd.MultiIndex(levels=[['non_touch']+[x for x in ind_perm]+['touch_ave','touch_var'], [x for x in X.columns.levels[1]]],
           labels=X.columns.labels,
           names=[u'first', u'second'])
        Z=X.copy();Z.columns=new_index
        data_perms.append(Z)
    data=pd.concat(data_perms,keys=X.columns)
    print(data)
    return(data)

def add_sample(X):
    return X

def add_var(X,n=1000):
    for j in xrange(n):
        mean=np.mean(X.iloc[:,1:])
        cov=np.diag(np.diag(np.cov(X.iloc[:,1:],rowvar=False,bias=True)))
        cov+=0.00001*np.eye(cov.shape[0],cov.shape[0])
        mean=np.zeros(cov.shape[0])
        rand=[]
        weight=0.6
        for i in xrange(X.shape[0]):
            rand.append(weight*np.random.multivariate_normal(mean,cov))
        Z=X.copy()
        Z.iloc[:,1:]=Z.iloc[:,1:]+np.array(rand)
        X.append(Z)

    return X
'''