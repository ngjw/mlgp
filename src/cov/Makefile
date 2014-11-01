include ../../make.inc

SINGLEOBJS = \
cov_sp.o \
cov_single_sp.o \
cov_composite_sp.o \
cov_aux_sp.o \
covSEiso_sp.o \
covSEard_sp.o \
covSum_sp.o \
covProd_sp.o \
covDummy_sp.o \

DOUBLEOBJS = \
cov_dp.o \
cov_single_dp.o \
cov_composite_dp.o \
cov_aux_dp.o \
covSEiso_dp.o \
covSEard_dp.o \
covSum_dp.o \
covProd_dp.o \
covDummy_dp.o \


ALLCOVS =

ifeq ($(SINGLE),1)
ALLCOVS+=$(SINGLEOBJS)
endif

ifeq ($(DOUBLE),1)
ALLCOVS+=$(DOUBLEOBJS)
endif

all: $(ALLCOVS)

%_sp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c $< -o $@

%_dp.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -DDOUBLE -fPIC -c $< -o $@

clean:
	rm -f *.o *.gcno
