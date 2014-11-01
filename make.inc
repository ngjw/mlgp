# SET THE COMPILER TO USE
CC=gcc
HAVELBFGS=1
COVERAGE=0

# 1 TO COMPILE SINGLE/DOUBLE PRECISION CODE, 0 TO OMIT
SINGLE=1
DOUBLE=1

BLASDIR=/usr/lib
LAPACKDIR=/usr/lib
MATHDIR=/lib/x86_64-linux-gnu

# uncomment these two lines and set the paths if libLBFGS is not
# installed at the default paths
#LBFGSINCLUDE=/usr/local/include
#LBFGSDIR=/home/local/lib

# SET THE DIRECTORY FOR THE LIBRARY
MLGPLIBDIR=/usr/local/lib

############################################
# CHANGE THE FIELDS BELOW AT YOUR OWN RISK #
############################################

CFLAGS = -std=c99 -O3

ifeq ($(COVERAGE),1)
CFLAGS += --coverage -g -fprofile-arcs -ftest-coverage
endif

CLIB += -L$(LAPACKDIR) -llapack
CLIB += -L$(BLASDIR)   -lblas
CLIB += -L$(MATHDIR)   -lm

DEFINE=

ifeq ($(SINGLE),1)
ifeq ($(DOUBLE),1)
DEFINE += -DBOTHPRECISION
endif
endif

ifeq ($(HAVELBFGS),1)
DEFINE += -DHAVELBFGS
INCLUDE += -I$(LBFGSINCLUDE)
CLIB += -L$(LBFGSDIR) -llbfgs
endif

LIBS = $(CLIB)
