CFLAGS = -std=c++17
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
release: main.cpp
	g++ $(CFLAGS) -o release -O3 main.cpp $(LDFLAGS)
