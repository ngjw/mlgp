include make.inc

all: libmlgp

libmlgp:
	cd src; make libmlgp.so

demo: libmlgp
	cd demo; make

clean:
	cd src; make clean;
	cd demo; make clean;
