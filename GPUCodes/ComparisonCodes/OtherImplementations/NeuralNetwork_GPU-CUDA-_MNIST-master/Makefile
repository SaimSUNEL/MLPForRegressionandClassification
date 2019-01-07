CC = nvcc -std=c++11
ARCH=-arch sm_20
NVOPTS=-O3 $(ARCH) -x cu
LIBS = -lcublas
ARGS=-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
BUILD = ./build
SOURCE = ./src

programs = $(BUILD)/main
objects = $(BUILD)/NN.o $(BUILD)/kernels.o $(BUILD)/functions.o

all: $(programs) copy

$(BUILD)/%.o: $(SOURCE)/%.cpp $(SOURCE)/%.h
	$(CC) $(ARGS) $(NVOPTS) -c $< -o $@

$(BUILD)/%.o: $(SOURCE)/%.cpp
	$(CC) $(ARGS) $(NVOPTS) -c $< -o $@

$(programs): %: %.o $(objects)
	$(CC) $(ARGS) $(LIBS) $(objects) $< -o $@

copy:
	cp $(programs) ./test/

.PHONY: clean copy
clean:
	rm -rf $(objects) $(programs) $(BUILD)/main.o *~
