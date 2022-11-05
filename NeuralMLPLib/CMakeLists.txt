#--------------------------------------------------------
# CMake for generating the NeuralMLPLib
#
# @author: Wild Boar
#
# Date Created: 2022-11-05
#--------------------------------------------------------

# Setup include directories
include_directories("../" "${LIBRARY_BASE}/NVLib")

# Create Library
add_library(NeuralMLPLib STATIC
    ArgUtils.cpp
    ScoreUtils.cpp
)
