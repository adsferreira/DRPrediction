import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "sans-serif"

elapsed_times = np.array([[14.5, 7.7, 4.6, 2.7, 1.5, 3.2],
                         [20.3, 10.9, 6.5, 4.1, 2.1, 4.5],
                         [83.1, 47.2, 27, 15.2, 8.5, 16.9],
                         [109.5, 70.2, 38.6, 22.2, 12, 20.9]])

min_elap_times = [[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                  [2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7],
                  [8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5],
                  [12, 12, 12, 12, 12, 12, 12, 12]]


colors = ['rv', 'b8', 'gd', 'kp']
lines = ['--', '--', '-', '-']
labels = ['TM-PC', 'TE-PC', 'Diamond', 'SC5']
x_index = np.linspace(0, 7, 8)

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)

for i in range(len(elapsed_times)):
    plt.plot(x_index[1:-1], elapsed_times[i], colors[i] + lines[i], linewidth = 3., markersize = 10, label = labels[i])
    plt.plot(np.arange(0, 8), min_elap_times[i], 'k:', linewidth = 1.5)
#     plt.setp(line, color=colors[i], linewidth = 9., label = labels[i])

x_ticks = ['', '1', '2', '4', '8', '16', '32', '']
plt.xticks(x_index, x_ticks)
#ax.set_xticks(x_index)
#ax.set_xticklabels(x_ticks)
plt.ylim(0, 111)  

ax.set_ylabel('Time(seconds)', fontsize=30)
ax.set_xlabel('Number of threads', fontsize = 30)
ax.tick_params(labelsize=24, length=6)

#legend
from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best', numpoints = 1, fontsize = 20)
    
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.tight_layout(pad=0.1, w_pad=0., h_pad=0.0)
plt.savefig('mpb_performance.pdf', dpi=300)
plt.show()    