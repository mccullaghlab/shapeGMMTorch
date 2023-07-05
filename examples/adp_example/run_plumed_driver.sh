source ./plumed_source_script.sh

# command line
#plumed --no-mpi driver --plumed plumed.dat --itrr ala_amber99_T300_2dMetad_wrapped.trr
#plumed --no-mpi driver --plumed plumed.dat --itrr ala_amber99_T300_2dMetad.trr


#plumed --no-mpi driver --plumed plumed_usgmm.dat --itrr ala_amber99_T300_2dMetad_wrapped.trr

#plumed --no-mpi driver --plumed plumed_wsgmm.dat --itrr ala_amber99_T300_2dMetad_wrapped.trr

# print the value of kt
plumed kt --temp 300.0 --units kcal/mol

# calculate the metadynamics weights from the simulation!
plumed driver --plumed 2d_metad.plumed.dat --itrr ala_amber99_T300_2dMetad.trr --kt 0.596161 

