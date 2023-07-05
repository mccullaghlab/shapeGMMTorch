#!/bin/bash
module purge
module load intel/19.1.2
module load cmake/3.18.4
module load openmpi/intel/4.0.5
module load cuda/11.1.74
plumed_dir=/scratch/work/hockygroup/software/plumed2_subarna/
plumed=$plumed_dir/bin/plumed
plumed_bin=$plumed_dir/bin

export PLUMED_KERNEL=$plumed_diir/lib/libplumedKernel.so
export LD_LIBRARY_PATH=$plumed_dir/lib:$LD_LIBRARY_PATH
export PATH=$plumed_bin:$PATH


