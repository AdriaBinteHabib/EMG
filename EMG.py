#taking dataset to the dataframe and merging all the files
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import os
import shutil
import sys

'''# --------------------------------------------------------
reorg_dir = sys.argv[1]
target_dir = sys.argv[2]
# ---------------------------------------------------------
for root, dirs, files in os.walk(reorg_dir):
    for name in files:
        subject = root+"/"+name
        n = 1; name_orig = name
        while os.path.exists(target_dir+"/"+name):
            name = "duplicate_"+str(n)+"_"+name_orig; n = n+1
        newfile = target_dir+"/"+name; shutil.move(subject, newfile)'''
        
reorg_dir = "/path/to/EMG_data_for_gestures-master"
target_dir = "/path/to/Directory" 
# ---------------------------------------------------------
for root, dirs, files in os.walk(reorg_dir):
    for name in files:
        subject = root+"/"+name
        n = 1; name_orig = name
        while os.path.exists(target_dir+"/"+name):
            name = "duplicate_"+str(n)+"_"+name_orig; n = n+1
        newfile = target_dir+"/"+name; shutil.move(subject, newfile)
        
'''path = './Directory/'
files = []
# r=root, d=directories, f = files
for r,d, f in os.walk(path):
    for file in f:
    
        if 'watch' in file:
            files.append(os.path.join(r, file))
frame=[]
Data=''  
for i in files:
    dataset= pd.read_csv(i)
    dataset.columns = ['Time','Channel1','Channel2','Channel3','Channel4',
                       'Channel5','Channel6','Channel7','Channel8','Class']
    print(dataset)
    DataFrame = pd.DataFrame(data=dataset)
    frame.append(DataFrame)
result = pd.concat(frame)'''