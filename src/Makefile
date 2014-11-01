include ../make.inc

INCLUDE += -I../include

SINGLEOBJS = \
predict_sp.o \
predict_cpu_sp.o \
likelihood_sp.o \
likelihood_cpu_sp.o \
blas_sp.o \
linalg_sp.o \
utils_sp.o \

DOUBLEOBJS = \
predict_dp.o \
predict_cpu_dp.o \
likelihood_dp.o \
likelihood_cpu_dp.o \
blas_dp.o \
linalg_dp.o \
utils_dp.o \

ifeq ($(HAVELBFGS),1)
SINGLEOBJS += train_sp.o
DOUBLEOBJS += train_dp.o
endif

ALLOBJS = COV MEAN

ifeq ($(SINGLE),1)
ALLOBJS+=$(SINGLEOBJS)
endif

ifeq ($(DOUBLE),1)
ALLOBJS+=$(DOUBLEOBJS)
endif


all: libmlgp.so

libmlgp.so: $(ALLOBJS)
	$(CC) $(CFLAGS) -shared *.o cov/*.o mean/*.o -o $(MLGPLIBDIR)/libmlgp.so

%_sp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c $< -o $@

%_dp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -DDOUBLE -fPIC -c $< -o $@


COV:
	cd cov; make

MEAN:
	cd mean; make

# compile the train function if the LBFGS Library is available
ifeq ($(HAVELBFGS),1)
train_sp.o: train.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c train.c -o train_sp.o

train_dp.o: train.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -DDOUBLE -fPIC -c train.c -o train_dp.o
endif

clean:
	rm -f *.o *.gcno
	cd cov; make clean;
	cd mean; make clean;
