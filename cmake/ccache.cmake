# written by dottik (yes i did write this garbage)

find_program(CCACHE_EXECUTABLE
        NAMES ccache ccache.exe
        HINTS
        "$ENV{ProgramFiles}/ccache"
        "C:/ProgramData/chocolatey/bin" # Chocolatey

        # macOS and Linux
        "/usr/local/bin"
        "/usr/bin"
        "/opt/homebrew/bin" # homebrew on apple
        "/opt/local/bin"    # macports
)

if (CCACHE_EXECUTABLE)
    message(STATUS "found ccache executable @ ${CCACHE_EXECUTABLE}")

    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "C++ compiler launcher")

    set(CCACHE_FOUND ON)
else ()
    message(WARNING "ccache not found. Building without compiler caching.")
    message(STATUS "ccache can be easily obtained @ https://ccache.dev/ and you will likely benefit from a hefty speed up in compilation afterward!")
endif ()
