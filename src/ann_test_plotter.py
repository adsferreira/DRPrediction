import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib as mpl
import csv
from matplotlib.ticker import FormatStrFormatter

def read_csv_file(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        return np.array(map(np.float64, list(reader)))
    
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "sans-serif"

file_names = ['test_sc5_pc_0_55.pdf', 'test_sc5_pc_1_35.pdf']
org_bands = read_csv_file('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/sc5/16_interpolated_points/dr_sc5_pc_dataset.csv')[621:,-6:]
mlp_bands = read_csv_file('tests/test_mlp_sc5_pc_outs.csv')
elm_bands = read_csv_file('tests/test_elm_sc5_pc_outs.csv')

k_ticks_square_major = ['$\Gamma$', 'X', 'M', '$\Gamma$']
k_ticks_diamond = ['X', 'U', 'L', '$\Gamma$', 'X', 'W', 'X']
k_ticks_cubic = ['$\Gamma$', 'R', 'X', 'M', '$\Gamma$']
nr_k_points = len(k_ticks_cubic)
interpolated_points = 16
total_k_points = ((nr_k_points - 1) * interpolated_points + nr_k_points)
k_index = np.arange(0, total_k_points, interpolated_points + 1)
ymax_pad = [0.02, 0.02]
nr_pcs = 2
ii = 0

for i in range(nr_pcs):
    ie = (i + 1) * total_k_points
    cur_org_bands = org_bands[ii:ie]
    cur_mlp_bands = mlp_bands[ii:ie]
    cur_elm_bands = elm_bands[ii:ie]
    ii = (i + 1) * total_k_points

    ymin = 0
    ymax = max(cur_org_bands[0]) + ymax_pad[i]
    #ymax = math.ceil(max(bands_tm[0]) * 10) / 10
    y_ticks = np.arange(ymin, ymax, 0.2)    
  
    fig = plt.figure()                                                               
    ax = fig.add_subplot(1,1,1) 

    modes = np.array(cur_org_bands).T
    min_vec = np.amin(modes, axis=1)
    max_vec = np.amax(modes, axis=1)
    nr_modes = modes.shape[0]

    for j in range(0, nr_modes - 1):
        if (min_vec[j + 1] - max_vec[j]) > 0:
            height = min_vec[j + 1] - max_vec[j]
            origin_y = max_vec[j]
            mid_gap_freq = origin_y + (height / 2)
            frac_gap_size = (height / mid_gap_freq) * 100
            # using fc with RGBA only affects the background and A does not affect the edge
            ax.add_patch(Rectangle (
                                (0.0, origin_y), # (x, y)
                                 total_k_points,       # width
                                 height,       # height
                                 #facecolor="gray",
                                 fc=(0, 0, 0, 0.1),
                                 linewidth=0.5,
                                 edgecolor='black',
                                 linestyle='dotted'
                                )
            )
        
            cx = total_k_points / 2.03
            cy = mid_gap_freq - 0.006

 
# plot with markers only
#     ax.plot(cur_org_bands, "ko", markersize=9.6, markerfacecolor='white', mew=1.3, label='MPB')
#     ax.plot(cur_elm_bands, 'k+', linewidth=4., markersize=10, mew=2, label='ELM')
#     ax.plot(cur_mlp_bands, 'k.', alpha=0.2, linewidth=4., markersize=13, label='MLP')

# plot with lines only
    ax.plot(cur_org_bands, "-k", linewidth=8.2, alpha=0.2, label='MPB')
    ax.plot(cur_elm_bands, '--r', linewidth=8.2, label='ELM')
    ax.plot(cur_mlp_bands, ':b', linewidth=8.2, label='MLP')
    
    plt.xlim(0, max(k_index))
    plt.ylim(0, ymax)   

    ax.set_xticks(k_index)
    ax.set_yticks(y_ticks)
    #ax.set_xticks(k_index_square_minor, minor=True) 
    ax.set_xticklabels(k_ticks_cubic)

    # put integer part at left
    ticklabs = ax.get_yticklabels()
    ax.set_yticklabels(ticklabs,ha='left')
    ax.yaxis.set_tick_params(pad=45) 
    
    ax.tick_params(labelsize=30)
    ax.tick_params(axis='y', which='major')
    ax.set_ylabel("$\omega a/2 \pi c$", fontsize=40)
    #ax.set_xlabel("Wave vector", fontsize=20)
    
    #ax.grid(which='both')                                                
    #ax.grid(which='minor', alpha=0.0)                                                
    ax.grid(which='major', axis='x', alpha=1.0) 
    
    # legend
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=25, numpoints=1, borderaxespad=0.3)
    
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.tight_layout(pad=0.1, w_pad=0., h_pad=0.0)
    plt.savefig(file_names[i], dpi=300)
    #plt.show()