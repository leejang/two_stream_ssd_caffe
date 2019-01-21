# compiler
CC :=g++
# include files
CFLAGS :=`pkg-config opencv --cflags` `pkg-config opencv --libs`
LDFLAGS := -lopencv_gpu
# compile all c++ files in dir
#SOURCES :=$(wildcard *.cpp)
SOURCES :=denseFlow_gpu.cpp
#SOURCES :=denseFlow.cpp
# C++ files .cpp removed file is executable
EXECUTABLE :=$(SOURCES:.cpp=)

INC_PATH :=-I/home/fan6/Program/opencv-2.4.13/build/include
LIB_PATH :=-L/home/fan6/Program/opencv-2.4.13/build/lib

all:$(EXECUTABLE)

$(EXECUTABLE):$(SOURCES)
	$(CC) $< $(INC_PATH) $(LIB_PATH) $(LDFLAGS) $(CFLAGS) -o $@

clean:
	rm -rf $(EXECUTABLE)
