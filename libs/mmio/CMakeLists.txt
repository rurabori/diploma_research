brr_add_library(mmio STATIC)

# uses posix calls, so widows suggests it's _<call> versions which aren't really
# necessary here.
target_compile_definitions(mmio PRIVATE _CRT_SECURE_NO_WARNINGS)
