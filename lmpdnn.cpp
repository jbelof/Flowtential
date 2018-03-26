// lmpdnn = umbrella driver to couple LAMMPS + tensorflow
//          for MD using DNN forces
// based on quest coupling example (SNL)
//
// Syntax: lmpdnn Niter in.lammps
//         Niter = # of MD timesteps
//         in.lammps = LAMMPS input script
//
// @2017, Jon Belof and Will Lowe

#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdint.h"

#include "many2one.h"
#include "one2many.h"
#include "files.h"
#include "memory.h"
#include "error.h"

#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

#include "lmppath.h"

#include QUOTE(LMPPATH/src/lammps.h)
#include QUOTE(LMPPATH/src/library.h)
#include QUOTE(LMPPATH/src/input.h)
#include QUOTE(LMPPATH/src/modify.h)
#include QUOTE(LMPPATH/src/fix.h)
#include QUOTE(LMPPATH/src/fix_external.h)

using namespace LAMMPS_NS;

void tensorflow_callback(void *, bigint, int, int *, double **, double **);

struct Info {
  int me;
  Memory *memory;
  LAMMPS *lmp;
};

/* ---------------------------------------------------------------------- */

int main(int narg, char **arg)
{
  int n;
  char str[128];

  // setup MPI

  MPI_Init(&narg,&arg);
  MPI_Comm comm = MPI_COMM_WORLD;

  int me,nprocs;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  Memory *memory = new Memory(comm);
  Error *error = new Error(comm);

  // command-line args
  if (narg != 3) error->all("Syntax: lmpdnn Niter in.lammps");

  int niter = atoi(arg[1]);
  n = strlen(arg[2]) + 1;
  char *lammps_input = new char[n];
  strcpy(lammps_input,arg[2]);
  n = strlen(arg[3]) + 1;

  // instantiate LAMMPS
  LAMMPS *lmp = new LAMMPS(0,NULL,MPI_COMM_WORLD);

  // create simulation in LAMMPS from in.lammps
  lmp->input->file(lammps_input);

  // make info avaiable to callback function
  Info info;
  info.me = me;
  info.memory = memory;
  info.lmp = lmp;

  // set callback to tensorflow inside fix external
  // this could also be done thru Python, using a ctypes callback
  int ifix = lmp->modify->find_fix("dyn_DNN");
  FixExternal *fix = (FixExternal *) lmp->modify->fix[ifix];
  fix->set_callback(tensorflow_callback,&info);

  // run LAMMPS for Niter
  // each time it needs forces, it will invoke tensorflow_callback
  sprintf(str,"run %d", niter);
  lmp->input->one(str);

  // clean up
  delete lmp;

  delete memory;
  delete error;

  delete [] lammps_input;

  MPI_Finalize();
}

/* ----------------------------------------------------------------------
   callback to tensorflow with atom IDs and coords from each proc
   invoke DNN to compute forces, load them into f for LAMMPS to use
   f can be NULL if proc owns no atoms
------------------------------------------------------------------------- */

void tensorflow_callback(void *ptr, bigint ntimestep,
		    int nlocal, int *id, double **x, double **f)
{
  int i,j;
  char str[128];

  Info *info = (Info *) ptr;

  // boxlines = strings that hold simulation box info
  char **boxlines = NULL;
  if (info->me == 0) {
    boxlines = new char*[3];
    for (i = 0; i < 3; i++) boxlines[i] = new char[128];
  }

  double boxxlo = *((double *) lammps_extract_global(info->lmp,"boxxlo"));
  double boxxhi = *((double *) lammps_extract_global(info->lmp,"boxxhi"));
  double boxylo = *((double *) lammps_extract_global(info->lmp,"boxylo"));
  double boxyhi = *((double *) lammps_extract_global(info->lmp,"boxyhi"));
  double boxzlo = *((double *) lammps_extract_global(info->lmp,"boxzlo"));
  double boxzhi = *((double *) lammps_extract_global(info->lmp,"boxzhi"));

  if (info->me == 0) {
    sprintf(boxlines[0],"%g %g %g\n", boxxlo, boxxhi, 0.);
    sprintf(boxlines[1],"%g %g %g\n", boxylo, boxyhi, 0.);
    sprintf(boxlines[2],"%g %g %g\n", boxzlo, boxzhi, 0.);
  }

  // xlines = string of coords for each atom
  int natoms;
  MPI_Allreduce(&nlocal,&natoms,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  Many2One *lmp2dnn = new Many2One(MPI_COMM_WORLD);
  lmp2dnn->setup(nlocal,id,natoms);

  /* create storage space for coords and coord strings */
  char **xlines = NULL;
  double **xtensorflow = NULL;
  if (info->me == 0) {
    xtensorflow = info->memory->create_2d_double_array(natoms,3,"lmpdnn:xtensorflow");
    xlines = new char*[natoms];
    for (i = 0; i < natoms; i++) xlines[i] = new char[128];
  }

  /* pull the coordinates and stuff them into xtensorflow */
  if (info->me == 0) lmp2dnn->gather(&x[0][0],3,&xtensorflow[0][0]);
  else lmp2dnn->gather(&x[0][0],3,NULL);

  /* write the coords into strings */
  for (i = 0; i < natoms; i++) {
    sprintf(xlines[i],"%g %g %g\n", xtensorflow[i][0],xtensorflow[i][1],xtensorflow[i][2]);
  }

  /* allocate space for forces and force strings */
  char **flines = NULL;
  double **ftensorflow = NULL;
  if (info->me == 0) {
    ftensorflow = info->memory->create_2d_double_array(natoms, 3, "lmpdnn:ftensorflow");
    flines = new char*[natoms];
    for (i = 0; i < natoms; i++) flines[i] = new char[128];
  }

  if (info->me == 0) {

    /* open coords file for DNN */
    FILE *fp = fopen("dnn.coords.dat", "w");

    /* write out number of atoms and box */
    //fprintf(fp, "%d\n", natoms);
    //fprintf(fp, "%lg %lg %lg %lg %lg %lg\n", boxxlo, boxxhi, boxylo, boxyhi, boxzlo, boxzhi);

    /* write out coords as ,-seperated csv */
    for(i = 0; i < natoms; i++)
      fprintf(fp, "%lg,%lg,%lg\n", xtensorflow[i][0], xtensorflow[i][1], xtensorflow[i][2]);

    /* close the coord file */
    fclose(fp);

    /* execute DNN */
    // polar h2o
    //system("python -W ignore ./keras.h2o.predict/pred.py ./keras.h2o.predict/distpolarreluscaled5ksteps.json ./keras.h2o.predict/distpolarreluscaled5ksteps.h5 dnn.coords.dat > /dev/null 2>&1");
    // LJ Cu
    system("python -W ignore ./keras.lj.predict/pred.py ./keras.lj.predict/distCuVelreluscaled1kstepsNVE_bignovelsigmoid.json ./keras.lj.predict/distCuVelreluscaled1kstepsNVE_bignovelsigmoid.h5 dnn.coords.dat > /dev/null 2>&1");

    /* open coords file for DNN */
    fp = fopen("dnn.forces.dat", "r");

    /* read in the forces */
    char line[1024];
    for(i = 0; i < natoms; i++) {
       fgets(flines[i], 1024, fp);
       sscanf(flines[i], "%lg %lg %lg", &ftensorflow[i][0], &ftensorflow[i][1], &ftensorflow[i][2]);
    }

    /* close the force file */
    fclose(fp);

  }

  // convert ftensorflow on one proc into f for atoms on each proc
  One2Many *dnn2lmp = new One2Many(MPI_COMM_WORLD);
  dnn2lmp->setup(natoms,nlocal,id);
  double *fvec = NULL;
  if (f) fvec = &f[0][0];
  if (info->me == 0) dnn2lmp->scatter(&ftensorflow[0][0],3,fvec);
  else dnn2lmp->scatter(NULL,3,fvec);

  // clean up
  // some data only exists on proc 0
  delete lmp2dnn;
  delete dnn2lmp;

  info->memory->destroy_2d_double_array(xtensorflow);
  info->memory->destroy_2d_double_array(ftensorflow);

  if (boxlines) {
    for (i = 0; i < 3; i++) delete [] boxlines[i];
    delete [] boxlines;
  }
  if (xlines) {
    for (i = 0; i < natoms; i++) delete [] xlines[i];
    delete [] xlines;
  }
  if (flines) {
    for (i = 0; i < natoms; i++) delete [] flines[i];
    delete [] flines;
  }


}
