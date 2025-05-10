from datetime import datetime

def count_15min_intervals(date_min, date_max):
    if isinstance(date_min, str):
        date_min = datetime.strptime(date_min, '%Y%m%d %H:%M:%S')
    if isinstance(date_max, str):
        date_max = datetime.strptime(date_max, '%Y%m%d %H:%M:%S')
    delta = date_max - date_min
    total_minutes = delta.total_seconds() / 60
    return int(total_minutes // 15) + 1