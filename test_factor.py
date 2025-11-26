
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from factor import factor
from analysis import *
from alpha101 import *
from gtja191 import *
from new_factor import *

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings


f=factor()
print(f.available_data())
         
#params={'ic_return_horizons': [5,20,60], 'ic_delay_periods': [1,2,5], 'ir_details': True}
params={'ir_details': True, 'n_groups':3, 'quantile':0.3}
factor_property={'universe':'top75pct','benchmark': 'cs500'}
#params={}

f=collective(params=params,factor_property=factor_property)
f.run()