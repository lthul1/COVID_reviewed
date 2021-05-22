import numpy as np
import data_loader as dl
import matplotlib.pyplot as plt
from datetime import datetime
from utility import *

def time_to_num(time_str):
    hh,mm,ss = map(float, time_str.split(':'))
    return ss + 60*(mm+60*hh)

a = datetime.now()
at = float(a.strftime("%S")) + 60*(float(a.strftime("%M")) + 60 * float(a.strftime("%H")))
# at = current_time.seconds + 60 * (current_time.minutes + 60 * current_time.hours)


lists = []
for j in range(int(10e7)):
    if j%10e6==0:
        b = datetime.now()
        print(str(b))
        bt = float(b.strftime("%f"))/10e5
        print(bt)

print(bt)

plt.show()

