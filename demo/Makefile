include ../make.inc

INCLUDE += -I../include

RPATHS=-Wl,-rpath=$(MLGPLIBDIR)

ifeq ($(HAVELBFGS),1)
RPATHS+=,-rpath=$(LBFGSDIR)
endif

space :=
space +=
RPATHSNS=$(subst $(space),,$(RPATHS))

demo: demo.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINE) demo.c -o demo.out -L$(MLGPLIBDIR) -lmlgp $(LIBS) $(RPATHSNS)

clean:
	rm *.o
