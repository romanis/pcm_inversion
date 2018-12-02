CXX      = /opt/rh/devtoolset-7/root/usr/bin/g++
#CXX	= tau_cxx.sh
#CXX	= icpc
SDIR     = .
KIDIR1	 = /opt/aci/sw/knitro/10.2.1/include
KIDIR2	 = /opt/aci/sw/knitro/10.2.1/examples/C++/include/
EIGEN	 = .#/storage/home/rji5040/work/eigen/
BOOSTI	 = .#/storage/work/rji5040/boost_1_65_1/

TASMANIANL = /storage/home/rji5040/work/Tasmanian_run/lib
TASMANIANI = /storage/home/rji5040/work/Tasmanian_run/include/
# parameters for gurobi
INC      = /storage/home/rji5040/work/gurobi752/linux64/include/
CARGS    = -m64 -g
CLIB     = -L/storage/home/rji5040/work/gurobi752/linux64/lib/ -lgurobi75
CPPLIB   = -L/storage/home/rji5040/work/gurobi752/linux64/lib/ -lgurobi_c++ 

MKL_FLAGS= -L"/opt/intel/compilers_and_libraries_2016.3.210/linux/tbb/lib/intel64/gcc4.4:/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64" -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
#-lgurobi75

#CXXFLAGS = -O2  -c -I$(SDIR) -I$(IDIR) -I$(KIDIR1) -I$(KIDIR2) -I$(EIGEN) -std=c++0x
CXXFLAGS = -O3 -std=c++14  -c -I$(SDIR)  -I$(KIDIR1) -I$(KIDIR2) -I$(EIGEN) -I. -I$(BOOSTI) -I$(INC) $(CARGS) -I$(TASMANIANI)   -fopenmp #-funroll-all-loops
#LDFLAGS  = -lm -L$(LDIR) -lscl   -fopenmp -rdynamic /opt/aci/sw/knitro/10.2.1/lib/libknitro.so -ldl -Wl,-rpath,/opt/aci/sw/knitro/10.2.1/lib
LDFLAGS  = -lm $(CPPLIB)   -fopenmp -rdynamic /opt/aci/sw/knitro/10.2.1/lib/libknitro.so.10.2.1 $(TASMANIANL)/libtasmaniansparsegrid.a -ldl -Wl,-rpath,/opt/aci/sw/knitro/10.2.1/lib $(MKL_FLAGS) 
#LDFLAGS  = -lm $(CPPLIB) -L$(BOOSTL) $(BOOSTL)/libboost_serialization.a -fopenmp -rdynamic 

main : main.o pcm_market_share.o matrix_inverse.o
	$(CXX) -o main main.o pcm_market_share.o matrix_inverse.o $(LDFLAGS)

main.o : $(SDIR)/main.cpp $(SDIR)/pcm_market_share.h 
	$(CXX) $(CXXFLAGS) -o main.o $(SDIR)/main.cpp
	
pcm_market_share.o : $(SDIR)/pcm_market_share.h $(SDIR)/pcm_market_share.cpp
	$(CXX) $(CXXFLAGS) -o pcm_market_share.o $(SDIR)/pcm_market_share.cpp
	
matrix_inverse.o: matrix_inverse.h matrix_inverse.cpp
	$(CXX) $(CXXFLAGS) -o matrix_inverse.o $(SDIR)/matrix_inverse.cpp


clean :
	rm -f *.o
	rm -f main

veryclean :
	rm -f *.o
	rm -f main 
