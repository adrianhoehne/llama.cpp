llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            moe-hot-cache/llama-moe-hot-cache-budget.cpp
            moe-hot-cache/llama-moe-hot-cache-builder.cpp
            moe-hot-cache/llama-moe-hot-cache.cpp
            moe-hot-cache/llama-moe-hot-cache-graph.cpp
            moe-hot-cache/llama-moe-hot-cache-parser.cpp
            moe-hot-cache/llama-moe-hot-cache-perf-json.cpp
            moe-hot-cache/llama-moe-hot-cache-perf.cpp
            moe-hot-cache/llama-moe-hot-cache-perf-nodes.cpp
            moe-hot-cache/llama-moe-hot-cache-planner.cpp
            moe-hot-cache/llama-moe-hot-cache-updater.cpp
            moe-hot-cache/llama-moe-hot-cache-weighting.cpp
            moe-hot-cache/llama-moe-hot-cache-worklist.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE . ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
