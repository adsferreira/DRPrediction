import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Bitstream Vera Sans"]
mpl.rcParams["font.family"] = "sans-serif"

def get_mse_vec(f_name):
    with open(f_name, 'rb') as f:
        reader = csv.reader(f)
        return list(reader)[0]

prefix = '/home/adriano/Projects/ANNDispersionRelation/ann_training/'

f_names = [prefix + '2d/square/tm/16_interpolated_points/best_loss_vec.csv',
           prefix + '2d/square/te/tests_new_db/16_interpolated_points/best_loss_vec.csv',
           prefix + '2d/square/dielectric_veins/with_material/te/ds/16_interpolated_points/loss_vec.csv',
           prefix + '2d/square/square_rods_veins/with_material/te/ds/16_interpolated_points/loss_vec.csv',
           prefix + '2d/triangular/tri_diel_rods/no_material/tm/ds/16_interpolated_points/loss_vec.csv',
           prefix + '3d/fcc/diamond/ds/16_interpolated_points/best_loss_vec.csv',
           prefix + '3d/fcc/diamond2/no_material/16_interpolated_points/best_loss_vec.csv',
           prefix + '3d/fcc/diamond3/no_material/ds/16_interpolated_points/loss_vec.csv',
           prefix + '3d/sc/sc5/ds/16_interpolated_points/best_loss_vec.csv',
           prefix + '3d/sc/spheres_cylindrical_veins/with_material/ds/16_interpolated_points/loss_vec.csv']

# mse of 2d phcs
mse_2d_sq_2c_phc = get_mse_vec(f_names[0])
mse_2d_sq_cs_phc = get_mse_vec(f_names[1])
mse_2d_sq_dv_phc = get_mse_vec(f_names[2])
mse_2d_sq_cc_phc = get_mse_vec(f_names[3])
mse_2d_tr_1c_phc = get_mse_vec(f_names[4])

# mse of 3d phcs
mse_3d_fc_as_phc = get_mse_vec(f_names[5])
mse_3d_fc_cb_phc = get_mse_vec(f_names[6])
mse_3d_fc_ds_phc = get_mse_vec(f_names[7])
mse_3d_sc_hc_phc = get_mse_vec(f_names[8])
mse_3d_sc_dc_phc = get_mse_vec(f_names[9])


type_phc = 2

# nr_datasets = 4
# f, axarr = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True)
# 
# axarr[0].plot(mse_tm_phc)
# axarr[1].plot(mse_te_phc)
# axarr[2].plot(mse_diamond)
# axarr[3].plot(mse_sc5)
# plt.show()

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1) 

if type_phc == 2:
    plt.plot(mse_2d_tr_1c_phc[0:200], 'r-',  linewidth = 7.5, alpha=0.4, label='2D-OneCyl-PhC')
    plt.plot(mse_2d_sq_2c_phc[0:200], 'g:',  linewidth = 7.5, alpha=0.5, label='2D-TwoCyl-PhC')
    plt.plot(mse_2d_sq_cs_phc[0:200], 'b-.', linewidth = 9.0, alpha=1.0, label='2D-ConStr-PhC')
    plt.plot(mse_2d_sq_dv_phc[0:200], 'k--', linewidth = 7.5, alpha=0.5, label='2D-ConVei-PhC')
    plt.plot(mse_2d_sq_cc_phc[0:200], 'y--', linewidth = 7.5, alpha=0.7, label='2D-CylVei-PhC')
    
else:
    plt.plot(mse_3d_fc_cb_phc[0:200], 'b-.', linewidth = 9.0, alpha=1.0, label='3D-CylBon-PhC')
    plt.plot(mse_3d_fc_ds_phc[0:200], 'k--', linewidth = 7.5, alpha=0.5, label='3D-DieSph-PhC')
    plt.plot(mse_3d_fc_as_phc[0:200], 'g:',  linewidth = 7.5, alpha=0.5, label='3D-AirSph-PhC')
    plt.plot(mse_3d_sc_hc_phc[0:200], 'r-',  linewidth = 7.5, alpha=0.4, label='3D-HolSph-PhC')
    plt.plot(mse_3d_sc_dc_phc[0:200], 'y--', linewidth = 7.5, alpha=0.7, label='3D-CylSph-PhC')
    
plt.ylim(0,0.50)

ax.set_xlabel("Iterations", fontsize=30)
ax.set_ylabel("MSE", fontsize=30)

ax.tick_params(labelsize=30)
ax.tick_params(axis='y', which='major')

ax.grid(which='minor', alpha=0.0)                                                
ax.grid(which='major', axis='y', alpha=1.0)
#legend
from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc=1, fontsize=28)
plt.tight_layout(pad=0.1, w_pad=0., h_pad=0.0)
plt.savefig('mlp_mse_vs_iterations_2DPhCs.pdf', dpi=300)
plt.show()