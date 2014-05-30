include ../make.inc

INCLUDE += -I../include

CPUOBJS = \
predict.o \
predict_cpu.o \
likelihood.o \
likelihood_cpu.o \
linalg.o \
utils.o \
COV \
MEAN \

ifeq ($(HAVELBFGS),1)
CPUOBJS += train.o
endif

ALLOBJS = $(CPUOBJS)


libmlgp.so: $(ALLOBJS)
	$(CC) $(CFLAGS) -shared *.o cov/*.o mean/*.o -o $(MLGPLIBDIR)/libmlgp.so

linalg.o: linalg.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c linalg.c

utils.o: utils.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c utils.c

predict.o: predict.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c predict.c

predict_cpu.o: predict_cpu.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c predict_cpu.c

likelihood.o: likelihood.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c likelihood.c

likelihood_cpu.o: likelihood_cpu.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c likelihood_cpu.c

COV:
	cd cov; make

MEAN:
	cd mean; make

# compile the train function if the LBFGS Library is available
ifeq ($(HAVELBFGS),1)
train.o: train.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c train.c
endif

clean:
	rm *.o;
	cd cov; make clean;
	cd mean; make clean;
