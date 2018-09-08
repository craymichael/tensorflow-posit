licenses(["notice"])


cc_library(
    name = "softposit",
    srcs = glob([
        "source/*.c",
    ]),
    hdrs = glob([
        "source/include/*.h",
        "build/Linux-x86_64-GCC/platform.h",
    ]),
    includes = ["source/include", "build/Linux-x86_64-GCC"],
    visibility = ["//visibility:public"],
)
