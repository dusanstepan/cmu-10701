#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from nbclassify import nbclassify
from math import sqrt

alpha = [sqrt(sqrt(10)) ** i for i in range(-20, 1)]
print alpha
accuracy = [nbclassify(alpha[i])[0] for i in range(len(alpha))]
#print len(alpha)
#accuracy = [nbclassify(alpha[i])[0] for i in range(2)]
print accuracy

fig, ax = plt.subplots()
p = ax.semilogx(alpha, accuracy, lw=2)
ax.grid('on')
ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')
plt.show()
