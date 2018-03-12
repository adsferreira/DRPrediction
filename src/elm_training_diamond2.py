from random_layer import RandomLayer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/diamond2/16_interpolated_points/')
    dataSet.read_csv_file('dr_diamond_pc_dataset.csv')  
    #print(len(dataSet.all_patterns[192:,:]))
    return dataSet

def set_patterns_test(dataSet):
    dataSet.split_inputs_and_targets(3)
    dataSet.create_patterns_set_for_testing2(927)
    #print(dataSet.targetsTesting)
    #dataSet.create_random_testing_set(2)

def scale_input_data(dataSet):
    # scale input data
    scaler = StandardScaler() 
    scaler.fit(dataSet.inputs)
    X_train = scaler.transform(dataSet.inputs)  
    X_test = scaler.transform(dataSet.inputsTesting) 
    
    return (X_train, X_test)

def train_ann(X_train, targets):
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=88, activation_func='multiquadric', alpha=0.63)),
                           ('lr', LinearRegression(fit_intercept=False))])

    #elmr = GenELMRegressor( hidden_layer = rl )
    #elmr = ELMRegressor(n_hidden=98,random_state=0, alpha=0.8)
    elm.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(elm, X_train))
    #print("training mse: ", tr_mse)
    # save model
    joblib.dump(elm, 'models/6_12_17/elm/diamond_pc/elm_XX_diamond_pc.pkl')
    
    return (elm, tr_mse)

def load_model():
    return joblib.load('models/6_12_17/elm/diamond_pc/elm_88_diamond_pc.pkl') 
        
def predict(elm, X_test):  
    return elm.predict(X_test)

def save_estimatives(outputs):
    with open('test_elm_diamond_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)

def plot_results(targets_test, mlp_results):
    k_ticks = ['X', 'U', 'L', '$\Gamma$', 'X', 'W', 'X']
    interpolated_points = 16
    total_nr_k_points = interpolated_points * (len(k_ticks) - 1) + len(k_ticks)
    k_index = np.arange(0, total_nr_k_points, interpolated_points + 1)
    
    solid_line = plt.plot(targets_test, "-")
    dotted_line = plt.plot(mlp_results, "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, total_nr_k_points - 1)
    plt.ylim(0, 1.4)
 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(True,  which="both")
    plt.ylabel("$\omega$a/2$\pi$c)$^2$", fontsize=30)
    plt.xlabel("K Vector", fontsize=30)
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 1)
 
    plt.tight_layout()
    #plt.savefig('pc_diamond_bs.pdf', dpi=300)
    plt.show()
    
    
if __name__ == '__main__':
    train = 1
    output = []
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
            
    if train:
        #te_mse = 1
        #while(te_mse > 2.98e-05):    
        mlp, mse = train_ann(X_train, ds.targets)
        #    outputs = predict(mlp, X_test)
        #    te_mse = mean_squared_error(ds.targetsTesting, outputs)
        #    print("test mse: ", te_mse)
    else:
        elm = load_model()
        outputs = predict(elm, X_test)
#         tr_mse = mean_squared_error(ds.targets, predict(elm, X_train))
#         te_mse = mean_squared_error(ds.targetsTesting, outputs)
#         print("training mse: ", tr_mse)
#         print("test mse: ", te_mse)
        
    #save_estimatives(outputs)
#     plot_results(ds.targetsTesting[0:103], outputs[0:103]) 
#     plot_results(ds.targetsTesting[103:], outputs[103:])
