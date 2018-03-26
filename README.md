# Flowtential

Numerical techniques for deep learning offer a route to the development of higher-fidelity (at the cost of complexity and computational expense) interatomic and intermolecular Hamiltonians.  This code base demonstrates a beta architecture for linking the molecular dynamic code LAMMPS with Google's deep neural network (DNN) library Tensorflow (by way of the python-based interface library Keras) in order to provide an interatomic potential through the force callback interface.  For further details, please see http://on-demand.gputechconf.com/gtc/2017/presentation/s7373-jon-belof-deep-neural-networkds-for-non-equilibrium-molecular.pdf

For any of this to be possible, you need to train a DNN and include the Tensorflow JSON file that will be used in a prediction step to provide forces for the MD code.  That means you need to know what the hell you are doing or none of this will be useful!

Flowtential is heavily-based on the very nice [QUEST](http://github.com/lammps/lammps/tree/master/examples/COUPLE/lammps_quest) code that uses the LAMMPS callback interface to call a DFT code for interatomic forces.

For use of this code, please cite J.L. Belof, E.W. Lowe and A. Hogan, "Deep neutral networks for non-equilibrium molecular dynamics", GPU Technology Conference (GTC), San Jose, CA, 2017


## Getting Started

This code was developed to utilize the open-source molecular dynamics code [LAMMPS](http://lammps.sandia.gov/) version of 3/21/2017 but will work with more recent versions.  LAMMPS needs to be compiled with libcouple and also needs to be linked as a library for external compilation (please see instructions in the LAMMPS source code for details).

Additional libraries that are needed include [Keras](https://keras.io), [Tensorflow](https://www.tensorflow.org) and various (more-or-less standard) python modules such as numba and numpy.

FFTW and MPI are also required for parallel simulation.


## Installing

Prerequisites are to compile and install FFTW, Tensorflow, Keras and LAMMPS with libcouple and the external API.  Next, the directory ./lammps.library includes commonly used routines in lammps, edit and run the Makefile there.

Edit lmppath.h and the Makefile to point to the location of your LAMMPS installation.

Open the main driver source, lmpdnn.cpp, and check the line that calls Keras.  Change any specifics of that call line to include your own predictor step and details of the DNN that has been trained.

Finally, run

$ make

to compile the main executable, lmpdnn.


## Running the example

The example included here is that of an atomic system described by the well-known Lennard-Jones potential.  A Tensorflow DNN has been trained over configurations coming from this exact potential; convolutional NN using RLU (dropout = 0.2) and 6 layers (2048/1024/512/256/512/2048) with input vectors containing coordinates and output vectors holding the forces.  The lmpdnn driver invokes an instantiation of LAMMPS and then gets called back via the line

\# set callback interface to retrieve forces from our trained DNN
fix             dyn_DNN all external pf/callback 1 1

from the input deck.  At this stage the coordinates get written to a file, the DNN gets invoked as a prediction step by way of Keras, and forces get written into a file.  The driver pulls the forces, copies them into the LAMMPS data structure and returns control back to LAMMPS.  This entire process happens each timestep and is incredibly slow.

Within a parallel instance, run the executable

$ ./lmpdnn <Niter> <in.lammps>

where Niter is the number of MD timesteps to be taken and in.lammps is the LAMMPS deck for your system.  Note that the only real requisite of the LAMMPS deck is that it use the callback interface for forces.


## Authors

* **Jon Belof** [jbelof@github](https://github.com/jbelof)  
* **Will Lowe**  

[google scholar](https://scholar.google.com/citations?user=gNrlNbwAAAAJ&hl=en)  
[research gate](https://www.researchgate.net/profile/Jon_Belof)  
[linkedin](http://www.linkedin.com/in/jbelof)  
[web profile](http://jbelof.academia.edu)  


## License

This project is licensed under the GNU General Public License v3, please see GPL_license.txt for details.


