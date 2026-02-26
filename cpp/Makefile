CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
TARGET = gpt2
TARGET_FAST = gpt2_fast
TARGET_KV = gpt2_kv

all: $(TARGET) $(TARGET_FAST) $(TARGET_KV)

$(TARGET): main.cpp model.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

$(TARGET_FAST): main.cpp model_simd.h
	$(CXX) $(CXXFLAGS) -DUSE_SIMD -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_FAST) main.cpp

$(TARGET_KV): main.cpp model_kvcache.h
	$(CXX) $(CXXFLAGS) -DUSE_KVCACHE -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_KV) main.cpp

clean:
	rm -f $(TARGET) $(TARGET_FAST) $(TARGET_KV)

.PHONY: all clean
