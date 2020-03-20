#include "tensor.h"
#include <stdio.h>
#include <fstream>

void load_file(char *data, const char *path, size_t size) {
    std::ifstream infile(path);
    infile.read(data, size);
}