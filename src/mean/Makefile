include ../../make.inc

MEANS = \
mean.o \
mean_aux.o \
meanOne.o \
meanConst.o \
meanLinear.o \

INCLUDE += -I../../include

all: $(MEANS)

mean.o: mean.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c mean.c

mean_aux.o: mean_aux.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c mean_aux.c

meanOne.o: meanOne.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c meanOne.c

meanConst.o: meanConst.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c meanConst.c

meanLinear.o: meanLinear.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) -fPIC -c meanLinear.c

clean:
	rm *.o;
