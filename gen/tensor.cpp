#include "tensor.h"
#include <stdio.h>
#include <fstream>

void load_file(void *data, const char *path, size_t size) {
    std::ifstream infile(path);
    infile.read(static_cast<char*>(data), size);
}
