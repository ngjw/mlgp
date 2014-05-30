# SET THE COMPILER TO USE
CC=gcc-4.4
HAVELBFGS=1
COVERAGE=0
DOUBLE=1

BLASDIR=/usr/lib
LAPACKDIR=/usr/lib
MATHDIR=/lib/x86_64-linux-gnu

LBFGSINCLUDE=/home/junwei/hroot/usr/include
LBFGSDIR=/home/junwei/hroot/usr/lib

# SET THE DIRECTORY FOR THE LIBRARY
MLGPLIBDIR=/home/junwei/mlgp/mlgp/lib

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

ifeq ($(HAVELBFGS),1)
DEFINE += -DHAVELBFGS
INCLUDE += -I$(LBFGSINCLUDE)
CLIB += -L$(LBFGSDIR) -llbfgs
endif

ifeq ($(DOUBLE),1)
DEFINE += -DDOUBLE
endif

LIBS = $(CLIB)
