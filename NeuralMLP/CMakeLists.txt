#--------------------------------------------------------
# CMake for generating the main executable
#
# @author: Wild Boar
#
# Date Created: 2022-11-05
#--------------------------------------------------------

# Setup the includes
include_directories("../")

# Create the executable
add_executable(NeuralMLP
    Engine.cpp
    Source.cpp
)

# Add link libraries                               
target_link_libraries(NeuralMLP NeuralMLPLib  NVLib ${OpenCV_LIBS})

# Copy Resources across
add_custom_target(resource_copy ALL
	COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/Resources ${CMAKE_BINARY_DIR}/NeuralMLP 
    DEPENDS ${MY_TARGET}
)
