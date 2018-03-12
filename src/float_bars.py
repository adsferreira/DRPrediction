# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "sans-serif"

def main():
    proportions = np.array([i for i in range(0, 9)])
    
    nr_datasets = 2
    f, axarr = plt.subplots(nr_datasets, sharex=True)
    
    path = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/'
    file_name = [path + 'tm/16_interpolated_points/freqs_pbgs.csv', path + 'te/tests_new_db/16_interpolated_points/freqs_pbgs.csv']
    pc_model = ['TM-polarized PhC', 'TE-polarized PhC']
    #path = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/'
    #file_name = [path + 'diamond2/16_interpolated_points/freqs_pbgs.csv', path + 'sc5/16_interpolated_points/freqs_pbgs.csv']
    #pc_model = ['Diamond PhC', 'SC5 PhC']
        
    i = 0
    
    while (i < nr_datasets):
        l_freqs, u_freqs = generate_data(file_name[i])
     
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
                        label=(str(j + 1) + 'º PBG').decode('utf8'))
            
           
        axarr[i].legend(loc =1, numpoints=1, ncol=3, columnspacing=0.7, handletextpad=0.25, borderpad=0.1, borderaxespad=0.5,fontsize=20)
        #axarr[i].grid(True, which='both', axis='y')   
        axarr[i].tick_params(labelsize=24, length=6)
        axarr[i].set_title(pc_model[i], fontsize=20)
        #axarr[i].set_yticklabels([0.3, 0.5, 0.7])
        axarr[i].set_yticklabels(['', '0.3', '','0.5', '', '0.7', '', '0.9'])
        axarr[i].set_ylim(0.2, max(u_freqs[0]) + 0.33)
                                
        i +=1
    
    #plt.setp(line_fem, color = 'k', linestyle='--', linewidth = 2.)  
    #plt.setp(line_ann, color = 'k', linestyle='--', linewidth = 2.)
    f.text(0, 0.5, "$\omega a/2 \pi c$", va='center', rotation='vertical', fontsize=30)
    
    plt.xlabel("Proportions (new PhCs)", fontsize = 20)
    x_ticks = ['0.5', '', '0.7', '', '0.9', '', '1.1', '', '1.3', '']
    plt.xticks(proportions, x_ticks)
    
    plt.tight_layout(rect=(0.05, 0, 1, 1))    
    plt.savefig('CF_PBG_2D.pdf', dpi = 700)
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