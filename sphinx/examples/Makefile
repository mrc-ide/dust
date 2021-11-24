SRC = $(wildcard *.cpp)
OUT = $(patsubst %.cpp,%.out,$(SRC))
BIN = $(patsubst %.cpp,%,$(SRC))

all: $(BIN) $(OUT)

%: %.cpp
	$(CXX) -O0 -I../../inst/include -o $@ $<

%.out: %
	./$< > $@

clean:
	$(RM) $(BIN) $(OUT)
