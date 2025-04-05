#include "fmt/core.h"
#include <iostream>

int main() {
    std::string message = "你好，世界！";
    auto formatted = fmt::format("Message: {}", message);
    fmt::println("{}", formatted);
    return 0;
}
