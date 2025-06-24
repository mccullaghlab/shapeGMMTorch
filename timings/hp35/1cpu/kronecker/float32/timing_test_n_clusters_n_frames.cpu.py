import numpy as np
import MDAnalysis as md
import torch
print(torch.__version__)
import time
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
import os
# import shapeGMM
from shapeGMMTorch import ShapeGMM

# scan parameters
device = torch.device("cpu")
torch.set_num_threads(1)
dtype = torch.float32
covar_type = "kronecker"
n_trials = 4
deltas = np.array([10,5,2,1])
component_array = np.array([1,2,3,4,5,6,7,8])
thresh = 1e-8
n_iter = 30

# read trajectory data
data_path = '../../'
prmtopFileName =  data_path + 'HP35.pdb'
trajFiles = [data_path + files for files in sorted(os.listdir(data_path)) if files.endswith('.dcd')]
coord = md.Universe(prmtopFileName,trajFiles)
sel = coord.select_atoms("name C CA N")
print("Number of frames:", coord.trajectory.n_frames)
print("Total number of atoms:", coord.atoms.n_atoms)
print("Number of selected atoms:", sel.n_atoms)

# read trajectory into numpy array
pos_traj = np.empty((coord.trajectory.n_frames,sel.n_atoms,3))
for ts in coord.trajectory:
    pos_traj[ts.frame] = sel.positions
# print number of frames
n_frames = coord.trajectory.n_frames // deltas
print("Frame array:", n_frames)
print("Component array:", component_array)
print("Log thresh:", thresh)
print("n_iter:", n_iter)
shape_gmm_times = np.empty((deltas.shape[0],5))
shape_gmm_times[:,0] = n_frames

for component in component_array:
    file_name = "v2.0_timing_" + covar_type + "_" +str(dtype) + "_" + str(component) + ".cpu.dat"
    for i, delta in enumerate(deltas):

       # in place shapeGMM
       elapsed_time = []
       scores = []
       for j in range(n_trials):
           start_time = time.process_time()
           sgmm = ShapeGMM(component,init_component_method='chunk',covar_type=covar_type,dtype=dtype,device=device,verbose=False,log_thresh=thresh, max_steps=n_iter)
           sgmm.fit(pos_traj[::delta])
           stop_time = time.process_time()
           elapsed_time.append(stop_time-start_time)
           scores.append(sgmm.score(pos_traj[::delta]))
       shape_gmm_times[i,1] = np.mean(elapsed_time)
       shape_gmm_times[i,2] = np.std(elapsed_time)
       shape_gmm_times[i,3] = np.mean(scores)
       shape_gmm_times[i,4] = np.std(scores)

    np.savetxt(file_name, shape_gmm_times)
