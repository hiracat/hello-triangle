CFLAGS = -std=c++17 -g -O0
LDFLAGS = -lglfw -lvulkan
VulkanTest: main.cpp
	g++ $(CFLAGS) -g -o VulkanTest main.cpp $(LDFLAGS)

vulkanmain: main.cpp
	g++ $(CFLAGS) -o vulkanmain main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	./VulkanTest

clean:
	rm -f VulkanTest
