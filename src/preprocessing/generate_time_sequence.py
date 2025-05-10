import pandas as pd
from datetime import datetime

def generate_15min_sequence(date_min, date_max):
    if isinstance(date_min, str):
        date_min = datetime.strptime(date_min, '%Y%m%d %H:%M:%S')
    if isinstance(date_max, str):
        date_max = datetime.strptime(date_max, '%Y%m%d %H:%M:%S')
    
    return pd.date_range(start=date_min, end=date_max, freq='15min')