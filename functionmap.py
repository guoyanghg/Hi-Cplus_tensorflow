import numpy as np
import matplotlib.pyplot as plt
import random

x=np.linspace(0,10,1000)

#y=[0.9,1.0,1.1,1.0,0.9,1.0,1.1,1.0,0.9,1.0,1.1,1.0,0.9]
y=[16.16+16*np.sin(i) for i in  x]
y2=[16 for i in  x]

plt.ylim((0, 33))
my_y_ticks = np.arange(0, 33, 1)
plt.yticks(my_y_ticks)
plt.plot(x,y,color='red')
plt.plot(x,y2)
plt.show()