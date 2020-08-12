# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "sans-serif"

def main():
    # it is necessary to check how many different proportions were used to form the datasets
    proportions = np.array([i for i in range(0, 9)])
    
    nr_datasets = 3
    # this does not work when 'nr_datasets' = 1. In order for it to work, one needs to remove the index i in 'axarr[i]' like 'axarr'
    f, axarr = plt.subplots(nr_datasets, sharex=True)
    
#     path_sq = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/'
#     path_tr = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/triangular/'
#     file_name = [path_tr + 'tri_diel_rods/no_material/tm/ds/16_interpolated_points/freqs_pbgs.csv',
#                  path_sq + 'tm/16_interpolated_points/freqs_pbgs.csv', 
#                  path_sq + 'te/tests_new_db/16_interpolated_points/freqs_pbgs.csv',
#                  path_sq + 'square_rods_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_8_9.csv',
#                  path_sq + 'square_rods_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_11_56.csv',
#                  path_sq + 'dielectric_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_13.csv',
#                  path_sq + 'dielectric_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_15.csv']
    #pc_model = ['2D-SQ-TwoCyl-PhC', '2D-SQ-ConStr-PhC']
    #pc_model = ['2D-TR-OneCyl-PhC']

    #pc_model = [r'2D-SQ-ConVei-PhC ($\epsilon=13$)', '2D-SQ-ConVei-PhC ($\epsilon=15$)']
    #pc_model = [r'2D-SQ-CylVei-PhC ($\epsilon=8.9$)', '2D-SQ-CylVei-PhC ($\epsilon=11.56$)']
    path = '/home/adriano/Projects/ANNDispersionRelation/ann_training/'
    file_name = [path + '3d/fcc/diamond3/no_material/ds/16_interpolated_points/freqs_pbgs.csv',
                 path + '2d/square/square_rods_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_8_9.csv',
                 path + '2d/square/square_rods_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_11_56.csv',
                 path + '2d/square/dielectric_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_13.csv',
                 path + '2d/square/dielectric_veins/with_material/te/ds/16_interpolated_points/freqs_pbgs_e_15.csv',
                 path + '2d/triangular/tri_diel_rods/no_material/tm/ds/16_interpolated_points/freqs_pbgs.csv',
                 path + '2d/square/tm/16_interpolated_points/freqs_pbgs.csv',
                 path + '2d/square/te/tests_new_db/16_interpolated_points/freqs_pbgs.csv',
                 path + '3d/fcc/diamond2/no_material/16_interpolated_points/freqs_pbgs.csv', 
                 path + '3d/sc/sc5/ds/16_interpolated_points/freqs_pbgs.csv',
                 path + '3d/fcc/diamond/ds/16_interpolated_points/freqs_pbgs.csv',
                 path + '3d/sc/spheres_cylindrical_veins/with_material/ds/16_interpolated_points/freqs_pbgs_e_13.csv',
                 path + '3d/sc/spheres_cylindrical_veins/with_material/ds/16_interpolated_points/freqs_pbgs_e_15.csv']
    
    pc_model = ['3D-DieSph-PhC',
                r'2D-CylVei-PhC ($\epsilon=8.9$)', 
                r'2D-CylVei-PhC ($\epsilon=11.56$)',
                r'2D-ConVei-PhC ($\epsilon=13$)',
                '2D-ConVei-PhC ($\epsilon=15$)',
                '2D-OneCyl-PhC',
                '2D-TwoCyl-PhC', 
                '2D-ConStr-PhC', 
                '3D-CylBon-PhC', 
                '3D-HolSph-PhC', 
                '3D-AirSph-PhC',
                r'3D-CylSph-PhC ($\epsilon=13$)',
                r'3D-CylSph-PhC ($\epsilon=15$)']
    #pc_model = ['FCC-A PhC', 'SC5 PhC']
    #pc_model = ['3D-FC-DieSph-PhC']
    #pc_model = [r'3D-SC-CylSph-PhC ($\epsilon=13$)', r'3D-SC-CylSph-PhC ($\epsilon=15$)']
    pols = ['', 'TE', 'TE', 'TE', 'TE', 'TM', 'TE', '', '', '', '', '', '', '', '']
    
    yticklabels = [['', '0.25', '0.30', '0.35', '0.40', '0.45'],
                   ['', '', '0.3', '', '0.4', '', '0.5', '', '0.6'],
                   ['', '', '0.3', '', '0.4', '', '0.5', '', '0.6'],
                   ['', '', '0.3', '', '0.4', '', '0.5', ''],
                   ['', '', '0.3', '', '0.4', '', '0.5', ''],
                   ['0.2', '', '0.3', '', '0.4', '', '0.5', ''],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],
                   ['', '', '0.3', '', '0.4', '', '0.5', ''],
                   ['', '', '0.3', '', '0.4', '', '0.5', ''],
                   ['', '0.3', '', '0.5', '', '0.7', '', '0.9'],]

        
    locs = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 2, 1, 1]
    #locs = [1, 1]
        
    i = 0
    step = 8
    while (i < nr_datasets):
        l_freqs, u_freqs = generate_data(file_name[i + step])
     
        pbgs = u_freqs - l_freqs
        cfs = l_freqs + (pbgs / 2)
        
        color = ['blue', 'gray']
        label=[r'$\bar\omega$', '']
               
        for j in range(0, len(pbgs)):
            
            mask = pbgs[j] > 0
                        
            axarr[i].plot(proportions[mask], cfs[j][mask], 'ko', alpha=1, markersize=8.5, label=label[j], linestyle='-', linewidth = 1.5)
            #axarr[i].plot(proportions, l_freqs[j], 'k_', alpha=1, markersize=25, mew=1.3, linestyle='--', linewidth = 1.5)
            #axarr[i].plot(proportions, u_freqs[j], 'k_', alpha=1, markersize=25, mew=1.3, linestyle='--', linewidth = 1.5)
            axarr[i].plot(proportions[mask], l_freqs[j][mask], 'k', alpha=1, markersize=7.5, linestyle=':', linewidth = 2.)
            axarr[i].plot(proportions[mask], u_freqs[j][mask], 'k', alpha=1, markersize=7.5, linestyle=':', linewidth = 2.)
          
            axarr[i].bar(proportions, pbgs[j], width = 0.35, color=color[j], bottom = l_freqs[j], alpha=0.3, align='center', 
                        label=(str(j + 1) + 'º ' + pols[i + step] + ' PBG').decode('utf8'))
            
           
        axarr[i].legend(loc =locs[i + step], numpoints=1, ncol=3, columnspacing=0.5, handletextpad=0.25, borderpad=0.1, borderaxespad=0.5,fontsize=19)
        #axarr[i].grid(True, which='both', axis='y')   
        axarr[i].tick_params(labelsize=24, length=6)
        axarr[i].set_title(pc_model[i + step], fontsize=20)
        #axarr[i].set_yticklabels([0.3, 0.5, 0.7])
        axarr[i].set_yticklabels(yticklabels[i + step])
        axarr[i].set_ylim(0.2, max(map(max, u_freqs)) + 0.2)
                                
        i +=1
            
    #plt.setp(line_fem, color = 'k', linestyle='--', linewidth = 2.)  
    #plt.setp(line_ann, color = 'k', linestyle='--', linewidth = 2.)
    f.text(0, 0.5, "$\omega a/2 \pi c$", va='center', rotation='vertical', fontsize=30)
    
    plt.xlabel("Proportions (variant PhCs)", fontsize = 20)
    x_ticks = ['0.5', '', '0.7', '', '0.9', '', '1.1', '', '1.3']
    #x_ticks = ['0.7', '', '0.9', '', '1.1', '', '1.3', '']
    #x_ticks = ['0.6', '0.8', '1.0', '1.2', '1.4']
    #x_ticks = ['0.7', '0.8', '1.0', '1.1', '1.2', '1.3']
    plt.xticks(proportions, x_ticks)
    
    plt.tight_layout(rect=(0.05, 0, 1, 1))    
    plt.savefig('3D-JLT_GapMap.pdf', dpi = 300)
    plt.show()


def generate_data(file_name):
    import csv
    
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        freqs_pbgs = np.array(list(reader))
        
#     print(freqs_pbgs)
    
    nr_phcs = len(freqs_pbgs) 
    nr_pbgs = len(freqs_pbgs[0]) / 2
    
    l_freqs = np.empty((nr_pbgs, nr_phcs))
    u_freqs = np.empty((nr_pbgs, nr_phcs))
     
    i = 0
    il = 0 # initial frequency index
    iu = 1 # final frequency index
    while (i < nr_pbgs):   
        l_freqs[i] = freqs_pbgs[:, il]    
        u_freqs[i] = freqs_pbgs[:, iu]
        i += 1
        il += 2
        iu +=2
            
    return(l_freqs, u_freqs)

if __name__ == '__main__':
    main()