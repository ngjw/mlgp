include ../../make.inc

INCLUDE += -I../../include

SINGLEOBJS = \
mean_sp.o \
mean_aux_sp.o \
meanOne_sp.o \
meanConst_sp.o \
meanLinear_sp.o \

DOUBLEOBJS = \
mean_dp.o \
mean_aux_dp.o \
meanOne_dp.o \
meanConst_dp.o \
meanLinear_dp.o \

MEANS =

ifeq ($(SINGLE),1)
MEANS+=$(SINGLEOBJS)
endif

ifeq ($(DOUBLE),1)
MEANS+=$(DOUBLEOBJS)
endif

all: $(MEANS)

%_sp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c $< -o $@

%_dp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -DDOUBLE -fPIC -c $< -o $@

clean:
	rm -f *.o *.gcno;
