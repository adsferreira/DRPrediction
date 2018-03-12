# this script generates dat files with the normalized frequencies
# of the following photonic crystals: TE 2D-PC, TM 2D-PC, diamond
# and SC5 (papers of H. Men et. al 2011, 2014, physics letters and
# optical express, respectively).

#December 23, 2017:
# added a 2D photonic crystal from MPB-MIT site:
# triangular lattice of rods in anisotropic dielectric and complete PBG
import os, fileinput, subprocess
import numpy as np

prefix = '/home/adriano/Projects/ANNDispersionRelation/ann_training/'

folders = [prefix + '2d/square/tm/', 
           prefix + '2d/square/te/tests_new_db/',
           prefix + '2d/square/complete/tm/', 
           prefix + '2d/triangular/te/',
           prefix + 'second_case_study/diamond2/16_interpolated_k_points/', 
           prefix + 'second_case_study/sc5/']

folder_id = 0
# find files with a specific format in a given directory
pc_ctl_files = []
pc_ctl_files += [dr_file for dr_file in os.listdir(folders[folder_id]) if dr_file.endswith('.ctl')]
pc_ctl_files.sort()

prop_to_edit = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '0.55', '1.5']
#prop_to_edit = ['0.4', '1.25']

for f_name in pc_ctl_files:
    name, file_extension = os.path.splitext(f_name)
    proportion_idx = name.index('p')
    proportion = name[proportion_idx + 1:]
    
    if (proportion in prop_to_edit):
#         vra = np.round(float(proportion), 1)
#               
#         f = fileinput.input(folders[folder_id] + f_name, inplace = True)
#         for line in f:
#             if 'interpolate' in line:
#                 line = line.replace("(set! k-points (interpolate 16 k-points))",
#                                     "(set! k-points (interpolate 16 k-points))")
    #         elif 'define-param r1' in line:
    #             proportion_idx = name.index('p')
    #             proportion_str = name[proportion_idx + 1:]
    #             proportion = float(proportion_str)
    #             r1 = 0.14 * proportion
    #             line = line.replace("(define-param r1 0.14)",
    #                                "(define-param r1 " + str(r1) + ")")
    #         elif 'define-param r2' in line:
    #             proportion_idx = name.index('p')
    #             proportion_str = name[proportion_idx + 1:]
    #             proportion = float(proportion_str)
    #             r2 = 0.36 * proportion
    #             line = line.replace("(define-param r2 0.36)",
    #                                "(define-param r2 " + str(r2) + ")")
    #         elif 'define-param r3' in line:
    #             proportion_idx = name.index('p')
    #             proportion_str = name[proportion_idx + 1:]
    #             proportion = float(proportion_str)
    #             r3 = 0.105 * proportion
    #             line = line.replace("(define-param r3 0.105)",
    #                                "(define-param r3 " + str(r3) + ")")
    #         elif 'resolution' in line:
    #             line = line.replace("(set-param! resolution 32)",
    #                                 "(set-param! resolution 16)")
#             elif 'define-param r' in line:
#                 radius = 0.28 * float(proportion)
#                 line = line.replace("(define-param r " + str(0.28 * np.round(float(proportion), 1)) + ")",
#                                     "(define-param r " + str(radius) + ")")
#                 
#             elif 'define-param d' in line:
#                 d = 0.042 * float(proportion)
#                 line = line.replace("(define-param d " + str(0.042 * np.round(float(proportion), 1)) + ")",
#                                     "(define-param d " + str(d) + ")")
#                         
#             print(line),
#         f.close()
         
        # call mpb to process the updated file
    
        p_mpb = subprocess.Popen(['mpb', folders[folder_id] + f_name], stdout = subprocess.PIPE)
        out_mpb, mpb_err = p_mpb.communicate()
        
        with open(folders[folder_id] + name + '.out', 'w') as f:
            f.writelines(out_mpb)
        
        # call mpb to process the updated file
        args = ['grep', 'tmfreqs', folders[folder_id] + name + '.out']
        p_grep = subprocess.Popen(args, stdout = subprocess.PIPE)
        out_grep, grep_err = p_grep.communicate()
        
        with open(folders[folder_id] + name + '.dat', 'w') as f:
            f.writelines(out_grep)
        
        print(f_name)