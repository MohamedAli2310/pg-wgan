import matplotlib.pyplot as plt
import numpy as np
import sys

# Using readlines()
file1 = open(sys.argv[1], 'r')
Lines = file1.readlines()
fake_critic = []
real_critic = []
distance = []
gen = []

param1 = []
param2 = []
param3 = []
param4 = []
param5 = []
param6 = []

prev=[]
# Strips the newline character
for line in Lines:
	words = line.split()
	prev = words
	if len(words) > 0 and words[0] == 'Epoch':
		fake_critic.append(words[11][:-1])
		real_critic.append(words[7][:-1])
		distance.append(words[13])
	if len(words) > 0 and words[0] == 'T,':
		param1.append(words[8][1:-1])
		param2.append(words[9][:-1])
		param3.append(words[10][:-1])
		param4.append(words[11][:-1])
		param5.append(words[12][:-1])
		param6.append(words[13][:-1])
		gen.append(words[-1])

"""
Graphing each parameter with a target line
"""

#param1
p = [float(x) for x in param1[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=1.25e-8, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("reco")
plt.legend()
plt.show()

#param2
p = [float(x) for x in param2[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=15000, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("N_anc")
plt.legend()
plt.show()

#param3
p = [float(x) for x in param3[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=2000, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("T_split")
plt.legend()
plt.show()

#param4
p = [float(x) for x in param4[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=0.05, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("mig")
plt.legend()
plt.show()

#param5
p = [float(x) for x in param5[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=9000, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("N1")
plt.legend()
plt.show()

#param6
p = [float(x) for x in param6[::100]]
x_axis = [x for x in range(len(p))]
plt.plot(x_axis, p, label='generated', color='g')
plt.axhline(y=5000, color='b', linestyle='-', label = 'real')
plt.xlabel("epochs(x100)")
plt.ylabel("N2")
plt.legend()
plt.show()

fake = [float(x) for x in fake_critic[::100]]
real = [-float(x) for x in real_critic[::100]]
dist = [float(x) for x in distance[::100]]
x_axis = [x for x in range(len(fake))]


plt.plot(x_axis, fake, label='fake critic loss')
plt.plot(x_axis, real, label='real critic loss')
plt.plot(x_axis, dist, label='total critic loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

g = [float(x) for x in gen[::100]]
x_axis = [x for x in range(len(g))]
plt.plot(x_axis, g)
plt.xlabel("epoch")
plt.ylabel("gen loss")
plt.show()
