
#! usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt




f = lambda x: (1 - x)/(1 - 2*x)

x = np.arange(0,0.2,.01)

plt.plot(x, f(x))



plt.show()





	



