#IMPORTANT: THIS CODE IS THOUGHT TO BE EXECUTED PROPERLY IN SPYDER, RUNNING EACH CELL IN ORDER
#AND SAVING THE FIGURES GENERATED MANUALLY
#Cells are separated using the string #%%

#FIRST CELL TO READ DATA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
plt.rc('font', family='Times New Roman', size=24)

model, bs , nGPUs = "base", "64", '16'


name= "<results>.json"

with open(name, 'r') as archivo:
    data=json.load(archivo)

Steps=np.array(data["Steps"]) #[[step number, ts, dur], ...]
Kernels=data["Kernels"] # {kernel: [[ts dur], ...], ...}



all_workers_step_times={}
for step, ts, dur in zip(Steps.T[0], Steps.T[1], Steps.T[2]):
    
    if step not in all_workers_step_times:
        all_workers_step_times[step]=[ts, ts+dur]
    
    if ts<all_workers_step_times[step][0]:
        all_workers_step_times[step][0]=ts
    
    if ts+dur>all_workers_step_times[step][1]:
        all_workers_step_times[step][1]=ts+dur

all_workers_kernel_times={}
spatha_times={}
for kernel_name, kernel_times in Kernels.items():
    
    for ts, dur in kernel_times:
        
        for step, times in all_workers_step_times.items():
            
            if step not in all_workers_kernel_times:
                all_workers_kernel_times[step]=[]
            
            if step not in spatha_times and "spatha" in kernel_name:
                spatha_times[step]=[]
                
            if times[0]<ts<times[1]:
                all_workers_kernel_times[step].append(dur)
                
            if times[0]<ts<times[1] and "spatha" in kernel_name:
                spatha_times[step].append(dur)            
                
#%%
fig, ax= plt.subplots(figsize=(9, 9))

tiempo_step=[]
for step, times in all_workers_step_times.items():
     tiempo_step.append([step, times[1] - times[0], times[0]])

tiempo_spatha=[]
for step, times in spatha_times.items():
    tiempo_spatha.append([step, sum(times)/int(nGPUs)])
        
tiempo_kernel=[]
for step, times in all_workers_kernel_times.items():
    tiempo_kernel.append([step, sum(times)/int(nGPUs)])
    

tiempo_step=np.array(tiempo_step).T
tiempo_spatha=np.array(tiempo_spatha).T
tiempo_kernel=np.array(tiempo_kernel).T

io=np.argsort(tiempo_step[2])
for v in (tiempo_step, tiempo_kernel, tiempo_spatha):
    for i, e in enumerate(v):
        v[i]=e[io]



def train_time(ax=ax):
    
    ax.plot((tiempo_step[2]-min(tiempo_step[2]))*1e-6/3600, tiempo_step[1]*1e-6, 'o', label = 'Step time')
    ax.plot((tiempo_step[2]-min(tiempo_step[2]))*1e-6/3600, tiempo_kernel[1]*1e-6, 'o', label = 'Mean use of GEMM kernels in a step')
    ax.plot((tiempo_step[2]-min(tiempo_step[2]))*1e-6/3600, tiempo_spatha[1]*1e-6, 'o', label = 'Mean use of SPATHA kernels in a step')
    ax.set(xlabel='Real time since first step (hours)',
           ylabel= 'Time (s)', title=f'BERT {model} with batch size {bs} in {nGPUs} GPUs'
                  )
    ax.legend(fontsize=12)
    

train_time()

fig.tight_layout()



#%%

change_index=[0]
change_index.extend(list(sorted(np.argsort(np.diff(tiempo_step[2]))[-3:]+1)))
change_index.append(len(tiempo_step[2]))

tick_labels=['dense', 'column_vector (64:6:8)', 'column_vector (64:4:8)', 'VENOM (64:2:8)']
# modify index labels depending of the configuration!!



fig, ax= plt.subplots(figsize=(16, 9))

def barras_unified(vector, color, label, idx=change_index, ax=ax):
    
    Means=[]
    Std=[]
    
    for i in range(len(idx)-1):
        
        state_times= list(vector[1][idx[i]:idx[i+1]])
        mean_time=np.mean(state_times)
        std= np.std(state_times)
        
        for t in state_times:
            if t > mean_time + 3*std: #Use this to erase ouliers in the statistics
                state_times.remove(t)
        
        mean_time_corrected=np.mean(state_times)
        Means.append(mean_time_corrected*1e-6)

        Std.append(np.std(np.array(state_times)*1e-6))
    ax.barh(list(range(len(idx)-1)), Means, tick_label=tick_labels,
             capsize=10, color=color, label=label)
        
    
barras_unified(tiempo_step ,  'blue', 'Step time')
barras_unified(tiempo_kernel, 'orange', 'GEMM time')
barras_unified(tiempo_spatha, 'red', 'SPATHA time')

ax.legend(bbox_to_anchor=(1, 0.9), fontsize=12)
ax.set(title=f'BERT {model} with batch size {bs} in {nGPUs} GPUs', xlabel="Time (s)")

fig.tight_layout()

    
#%%

min_t= np.inf
for k_name, k_times in Kernels.items():
    k=np.array(k_times).T
    if min_t>min(k[0]):
        min_t=min(k[0])
        
fig, ax= plt.subplots(figsize=(16, 9))

for k_name, k_times in Kernels.items():
    k=np.array(k_times).T
    ax.plot((k[0]-min_t)*1e-6/3600, k[1]*1e-3, 'bo', alpha=0.5)
    if "spatha" in k_name:
        ax.plot((k[0]-min_t)*1e-6/3600, k[1]*1e-3, 'ro', alpha=0.5)

ax.set(title=f'BERT {model} with batch size {bs} in {nGPUs} GPUs',
       xlabel="Time (hours)", ylabel="Kernel time (ms)")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Spatha Kernels',
           markerfacecolor='r', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='GEMM Kernels',
           markerfacecolor='b', markersize=10)
]

ax.legend(handles=legend_elements)

#%%CELL TO CLASIFY KERNELS IN A DICTIONARY





kernels_in_each_step={}

for step, times in all_workers_step_times.items():
    kernels_in_each_step[step]={}

for kernel_name, all_kernel_calls in Kernels.items():
    for kernel_time in all_kernel_calls:
        for step, step_time in all_workers_step_times.items():
            if step_time[0]<kernel_time[0]<step_time[1]:
                if kernel_name not in kernels_in_each_step[step]:
                    kernels_in_each_step[step][kernel_name]=[kernel_time[1]]
                else:
                    kernels_in_each_step[step][kernel_name].append(kernel_time[1])


def short_kernel(kernels: list):
    output={}
    for k in kernels:
        if "ampere" in k:
            s='ampere_'
            k_splited=k.split('_')
            for e in k_splited:
                if 'x' in e:
                    if [i.isnumeric() for i in e.split('x')]==[True, True]:
                        s+=e+'_'
            s+=k_splited[-1]
            output[k]=s
        elif 'cutlass' in k:
            s='cutlass_'
            k_splited=k.split('_')
            for e in k_splited:
                if 'x' in e:
                    if [i.isnumeric() for i in e.split('x')]==[True, True]:
                        s+=e+'_'
                if e=='nt' or e=='tn' or e=='nn':
                    s+=e
                    output[k]=s
                    break
        elif 'sddmm' in k:
            output[k]='spatha sddmm'
        elif 'spmm' in k:
            output[k]='spatha spmm'
        elif 'sm80_xmma_gemm' in k:
            s='sm80_xmma_'
            k_splited=k.split('_')
            for e in k_splited:
                if e=='nt' or e=='tn' or e=='nn':
                    form=e
                if [i.isnumeric() for i in e.strip('tilesize').split('x')]==[True, True, True]:
                    s+=e.strip('tilesize')+"_"+form
            output[k]=s
                
                
                
        else:
            print(k)
    return output


short_kernel_dict=short_kernel(sorted(Kernels.keys()))
#%%
plt.close('all')
kernel_keys = list(reversed(sorted(Kernels.keys())))
cmap = plt.get_cmap('tab20', 20)
kernel_color_dict = {kernel: cmap(i) for i, kernel in enumerate(kernel_keys)}

kernels_in_each_step_sorted=dict(sorted(kernels_in_each_step.items(), key=lambda item: item[0]))
for step, kernel_times in kernels_in_each_step_sorted.items():
    for k in kernel_keys:
        if k not in kernel_times:
            kernel_times[k]=[0]
tit=-1
for step, kernel_data_non_sorted in kernels_in_each_step_sorted.items():
    if step not in tiempo_step[0][change_index[:-1]]:
        continue
    data_quesito={}
    std={}
    kernel_data = dict(sorted(kernel_data_non_sorted.items(),  key=lambda item: item[0]))
    for kernel_name, times in kernel_data.items():
        data_quesito[kernel_name]=np.mean(times)
        std[kernel_name]=np.std(times)
    fig, ax= plt.subplots(figsize=(16, 8))
    ax.barh(list(range(0, len(list(data_quesito.values())))),
           list(data_quesito.values()),
           tick_label=[short_kernel_dict[x] for x in data_quesito.keys()],
           color=[kernel_color_dict[s] for s in data_quesito.keys()])
    tit+=1
    ax.set(title=tick_labels[tit], xlabel=r'Kernel mean time ($\mu$s)')
    fig.tight_layout()

#%%      

min_t= np.inf
for k_name, k_times in Kernels.items():
    k=np.array(k_times).T
    if min_t>min(k[0]):
        min_t=min(k[0])
        
fig, ax= plt.subplots(figsize=(16, 9))

for k_name, k_times in Kernels.items():
    k=np.array(k_times).T
    ax.plot((k[0]-min_t)*1e-6/3600, k[1]*1e-3, 'o', color=kernel_color_dict[k_name], alpha=0.5)
    # if "spatha" in k_name:
    #     ax.plot((k[0]-min_t)*1e-6/3600, k[1]*1e-3, 'o', color=kernel_color_dict[k_name], alpha=0.5)

ax.set(title=f'BERT {model} with batch size {bs} in {nGPUs} GPUs', xlabel="Time (hours)", ylabel="Kernel time (ms)")

#Add the labels with these lines
#legend_elements = [Line2D([0], [0], marker='o', color='w', label=short_kernel_dict[k_name], markerfacecolor=kernel_color_dict[k_name], markersize=10) for k_name in sorted(Kernels.keys())]
# ax.legend(handles=legend_elements, fontsize=10)   
