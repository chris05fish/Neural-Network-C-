INCLUDE_DIRS = 
LIB_DIRS = 
CPP = g++

CDEFS=
LIBS=

//CPPFLAGS= -g -Wall -fopenmp $(INCLUDE_DIRS) $(CDEFS)

PRODUCT= run

HFILES= ann.h
CPPFILES= ann.cpp main.cpp

SRCS= ${HFILES} ${CPPFILES}
OBJS= ${CPPFILES:.cpp=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

run:	$(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $(OBJS)

ann.o:	ann.cpp ann.h
	$(CPP) $(CPPFLAGS) -c ann.cpp

main.o:	main.cpp
	$(CPP) $(CPPFLAGS) -c main.cpp
