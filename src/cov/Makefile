include ../../make.inc

COVSCPU = \
cov.o \
cov_single.o \
cov_composite.o \
cov_aux.o \
covSEiso.o \
covSEard.o \
covSum.o \
covProd.o \
covDummy.o \


ALLCOVS = $(COVSCPU)

all: $(ALLCOVS)

cov.o: cov.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c cov.c

cov_single.o: cov_single.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c cov_single.c

cov_composite.o: cov_composite.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c cov_composite.c

cov_aux.o: cov_aux.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c cov_aux.c

covSEiso.o: covSEiso.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c covSEiso.c

covSEard.o: covSEard.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c covSEard.c

covSum.o: covSum.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c covSum.c

covProd.o: covProd.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c covProd.c

covDummy.o: covDummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c covDummy.c

clean:
	rm *.o;
