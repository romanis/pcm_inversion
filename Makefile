#CXX      = /opt/AMD/aocc-compiler-3.1.0/bin/clang++
CXX	= g++
SDIR     = .
NLOPTI 	 = /opt/NLopt/include/
NLOPTL   = /opt/NLopt/lib/
EIGEN	 = /usr/include/eigen3/

TASMANIANL = /usr/local/TASMANIAN/lib
TASMANIANI = /usr/local/TASMANIAN/include/
# parameters for gurobi
INC      = /usr/include/
CARGS    = -m64 -g
CLIB     = -L/storage/home/rji5040/work/gurobi752/linux64/lib/ -lgurobi75
CPPLIB   = -L. -L$(TASMANIANL)

MKL_FLAGS= -L"/opt/intel/compilers_and_libraries_2016.3.210/linux/tbb/lib/intel64/gcc4.4:/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64" -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
#-lgurobi75

#CXXFLAGS = -O2  -c -I$(SDIR) -I$(IDIR) -I$(KIDIR1) -I$(KIDIR2) -I$(EIGEN) -std=c++0x
CXXFLAGS = -O3 -std=c++14  -c -I$(SDIR)  -I$(EIGEN) -I. -I$(BOOSTI) -I$(INC) $(CARGS) -I$(TASMANIANI)   -fopenmp #-funroll-all-loops
#LDFLAGS  = -lm -L$(LDIR) -lscl   -fopenmp -rdynamic /opt/aci/sw/knitro/10.2.1/lib/libknitro.so -ldl -Wl,-rpath,/opt/aci/sw/knitro/10.2.1/lib
KNITROL = -rdynamic /opt/aci/sw/knitro/10.2.1/lib/libknitro.so.10.2.1 -Wl,-rpath,/opt/aci/sw/knitro/10.2.1/lib -L/storage/home/rji5040/work/gurobi752/linux64/lib/ -lgurobi_c++ 
LDFLAGS  = -lm $(CPPLIB)   -fopenmp  -ltasmaniansparsegrid -ldl #$(MKL_FLAGS) $(KNITROL)
#LDFLAGS  = -lm $(CPPLIB) -L$(BOOSTL) $(BOOSTL)/libboost_serialization.a -fopenmp -rdynamic 

main : main.o pcm_market_share.o matrix_inverse.o
	$(CXX) -o main main.o pcm_market_share.o matrix_inverse.o $(LDFLAGS)

main.o : $(SDIR)/main.cpp $(SDIR)/pcm_market_share.h 
	$(CXX) $(CXXFLAGS) -o main.o $(SDIR)/main.cpp
	
pcm_market_share.o : $(SDIR)/pcm_market_share.hpp $(SDIR)/pcm_market_share.cpp
	$(CXX) $(CXXFLAGS) -o pcm_market_share.o $(SDIR)/pcm_market_share.cpp -fPIC
	
pcm_market_share.so: pcm_market_share.o 
	$(CXX) -shared -o libpcm_market_share.so pcm_market_share.o -lm

matrix_inverse.o: matrix_inverse.h matrix_inverse.cpp
	$(CXX) $(CXXFLAGS) -o matrix_inverse.o $(SDIR)/matrix_inverse.cpp

test_pcm.o: pcm_market_share.so test_pcm.cpp
	$(CXX) -o test_pcm.o -c test_pcm.cpp $(CXXFLAGS) 

test_pcm: test_pcm.o pcm_market_share.so
	$(CXX) -o test_pcm test_pcm.o -L. -lpcm_market_share -fopenmp -L$(TASMANIANL) -ltasmaniansparsegrid 

clean :
	rm -f *.o
	rm -f main

veryclean :
	rm -f *.o
	rm -f main 
