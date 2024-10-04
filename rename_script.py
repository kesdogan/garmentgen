import os 
import sys 
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

datapath = Path('extensive_test_200_241001-00-46-20/default_body')
folders = next(os.walk(datapath))[1] 

folders = list(filter(lambda x: x.startswith('rand'), folders))
for folder in folders: 
     elements = folder.split('_')
     if len(elements[-1]) == len('2F8JG2OGMD'): 
            
        new_l = elements[0:1] + elements[-1:] + elements[1:-1]
        
        new_name = '_'.join(new_l)
        os.rename(datapath/folder, datapath/new_name)