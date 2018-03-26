# Makefile for MD with DNN forces via LAMMPS <-> tensorflow coupling
#
# @2017, Jon Belof and Will Lowe

SHELL = /bin/sh

# System-specific settings

LAMMPS =	/users/jbelof/lammps-31Mar17.DNN

CC =		mpicxx
CCFLAGS =	-g -O -I./lammps.library
DEPFLAGS =	-M
LINK =		mpicxx
LINKFLAGS =	-g -O -L./lammps.library -L${LAMMPS}/src -L/users/jbelof/FFTW/lib
USRLIB =	-lcouple -llammps_mpi
SYSLIB =	-lfftw -lmpi -lpthread
ARCHIVE =	ar
ARFLAGS =	-rc
SIZE =		size

# Files

EXE = 	lmpdnn
SRC =	$(wildcard *.cpp)
INC =	$(wildcard *.h)
OBJ = 	$(SRC:.cpp=.o)

# Targets

$(EXE):	$(OBJ)
	$(LINK) $(LINKFLAGS) $(OBJ) $(USRLIB) $(SYSLIB) -o $(EXE)
	$(SIZE) $(EXE)

clean:
	rm -f $(EXE) *.o

# Compilation rules

%.o:%.cpp
	$(CC) $(CCFLAGS) -c $<

%.d:%.cpp
	$(CC) $(CCFLAGS) $(DEPFLAGS) $< > $@

# Individual dependencies

DEPENDS = $(OBJ:.o=.d)
include $(DEPENDS)
