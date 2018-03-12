import matplotlib.pyplot as plt

circle1 = plt.Circle((-0.3, -0.3), 0.8, color='k')
circle2 = plt.Circle((-0.3, -0.3), 0.65, color='w')
#circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
ax.set_aspect('equal')
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_artist(circle1)
ax.add_artist(circle2)

plt.tight_layout()
plt.show()