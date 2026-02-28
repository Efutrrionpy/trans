CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
TARGET = gpt2
TARGET_FAST = gpt2_fast
TARGET_KV = gpt2_kv
TARGET_KV2 = gpt2_kv2
TARGET_Q8 = gpt2_q8

all: $(TARGET) $(TARGET_FAST) $(TARGET_KV) $(TARGET_KV2) $(TARGET_Q8)

$(TARGET): main.cpp model.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

$(TARGET_FAST): main.cpp model_simd.h
	$(CXX) $(CXXFLAGS) -DUSE_SIMD -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_FAST) main.cpp

$(TARGET_KV): main.cpp model_kvcache.h
	$(CXX) $(CXXFLAGS) -DUSE_KVCACHE -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_KV) main.cpp

$(TARGET_KV2): main.cpp model_kvcache_v2.h
	$(CXX) $(CXXFLAGS) -DUSE_KVCACHE_V2 -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_KV2) main.cpp

$(TARGET_Q8): main.cpp model_q8_kv.h
	$(CXX) $(CXXFLAGS) -DUSE_Q8 -DACCELERATE_NEW_LAPACK -framework Accelerate -o $(TARGET_Q8) main.cpp

clean:
	rm -f $(TARGET) $(TARGET_FAST) $(TARGET_KV) $(TARGET_KV2) $(TARGET_Q8)

.PHONY: all clean
