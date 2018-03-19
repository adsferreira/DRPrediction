import subprocess, re, csv
import numpy as np

class DatasetGenerator():
    def __init__(self, pol, n_phcs, f_name, phc_param_val=dict()):
        self.polarization = pol
        self.nr_phcs = n_phcs
        self.props = np.empty(n_phcs+1)
        self.f_name = f_name
        self.phc_param_val = phc_param_val
        self.is_linear = 0
        
    def set_linear_proportions(self, values):
        self.props[0] = 1
        self.props[1:] = np.array(values)
        self.is_linear = 1   
             
    def set_randn_proportions(self):
        self.props[0] = 0
        self.props[1:] = np.random.normal(size=self.nr_phcs) / 100
        self.is_linear = 0
        
    def generate_ds(self, ds_f_name):
        dataset = self.get_patterns()
        
        with open(ds_f_name, 'wb') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(dataset)
        
    def get_patterns(self):
        patterns = []
                
        if (self.is_linear):        
            for prop in self.props[1:]:
                param_values = []
                command = ['mpb']
                # create parameter list and store current values
                for key, value in self.phc_param_val.iteritems():
                    command.append(key + '=' + str(value * prop))
                    param_values.append(str(value * prop))
                
                # add ctl file to the mpb argument                                        
                command.append(self.f_name)
                # run mpb
                p_mpb = subprocess.Popen(command, stdout = subprocess.PIPE)
                out_mpb, mpb_err = p_mpb.communicate()
                
                if mpb_err == None:
                    disp_relation = self.grep_freqs(out_mpb, self.polarization)
                    current_patterns = self.concat_patterns(param_values, disp_relation)
                    #print(current_patterns)
                    patterns.append(current_patterns)
                else:
                    print('Something went wrong with MPB execution.\n')
        else:
            for prop in self.props:
                param_values = []
                command = ['mpb']
                # create parameter list and store current values
                for key, value in self.phc_param_val.iteritems():
                    command.append(key + '=' + str(value + prop))
                    param_values.append(str(value + prop))
                 
                # add ctl file to the mpb argument                                       
                command.append(self.f_name)
                # run mpb
                p_mpb = subprocess.Popen(command, stdout = subprocess.PIPE)
                out_mpb, mpb_err = p_mpb.communicate()
                
                if mpb_err == None:
                    disp_relation = self.grep_freqs(out_mpb, self.polarization)
                    current_patterns = self.concat_patterns(param_values, disp_relation)
                    #print(current_patterns)
                    patterns.append(current_patterns)
                else:
                    print('Something went wrong with MPB execution.\n')
       
        vra = np.array(patterns)  
        patterns = vra.reshape(vra.shape[0] * vra.shape[1], vra.shape[2])

        return patterns
                
    def grep_freqs(self, out_mpb, polarization):
        disp_relation = []
        aux = re.split('\n+', out_mpb)
        
        if polarization == 'tm':
            for line in aux:
                if 'tmfreqs' in line:
                    disp_relation.append(re.split(',', line))
        elif polarization == 'te':
            for line in aux:
                if 'tefreqs' in line:
                    disp_relation.append(re.split(',', line))
        elif polarization == 'all':
            for line in aux:
                if 'freqs' in line:
                    disp_relation.append(re.split(',', line))
        else:
            print("neither tm, te nor both polarizations were chosen.\n")
        
        disp_relation = np.array(disp_relation)[1:,2:]
        return disp_relation
       
    def concat_patterns(self, param_values, disp_relation):    
        patterns = []
         
        for freqs in disp_relation:
            patterns.append(np.concatenate([param_values, freqs]).astype(np.float))
         
        return patterns
             
        
prefix = '/home/adriano/Projects/ANNDispersionRelation/ann_training/'
 
folders = [prefix + '2d/square/tm/', 
           prefix + '2d/square/te/tests_new_db/',
           prefix + '2d/square/complete/tm/', 
           prefix + '2d/triangular/isotropic/air-columns/all_modes/random/',
           prefix + 'second_case_study/diamond2/16_interpolated_k_points/', 
           prefix + 'second_case_study/sc5/']
 
folder_id = 3
prefix = folders[folder_id]
f_name = prefix + 'one_rod_r_0.48.ctl'
polarization = 'all'
linear_props = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 0.5]
phc_param_val = {'r': 0.48}

ds_gen = DatasetGenerator(polarization, 7, f_name, phc_param_val)
ds_gen.set_randn_proportions()
ds_gen.generate_ds(prefix + '4_interpolated_points/tri_one_rod_ds.csv')