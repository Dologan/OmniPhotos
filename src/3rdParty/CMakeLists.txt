set(MODULE_NAME 3rdParty)


## Set up 3rdParty library ----------------------------------------------------

add_library(${MODULE_NAME}
  cxxopts.hpp
  Eigen.hpp
  fs_std.hpp
  json/jsoncpp.cpp
  json/json.h
  json/json-forwards.h
  
  openvr/files/compat.h
  openvr/files/Matrices.h
  openvr/files/Matrices.cpp
  openvr/files/Vectors.h
  
  stb_image.h
  tinyfiledialogs/tinyfiledialogs.cpp
  tinyfiledialogs/tinyfiledialogs.h
  tiny_obj_loader.cc
  tiny_obj_loader.h
  ${GL3W_FILES}
  ${OPENVR_FILES}
)

target_include_directories(${MODULE_NAME}
  PUBLIC
    ${GL3W_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

## Adding headers to the public precompiled header.
## NB. For some reason, CMake produces both C and C++ PCH,
##     and the former don't compile. -- CR 2020-02-29
#target_precompile_headers(${MODULE_NAME} PUBLIC
#  <3rdParty/cxxopts.hpp>
#  <3rdParty/fs_std.hpp>
#  <3rdParty/json/json.h>
#  <3rdParty/json/json-forwards.h>
#  <GL/gl3w.h>
#)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(${MODULE_NAME} PUBLIC OpenGL::GLX ${CMAKE_DL_LIBS})
endif()

set_property(TARGET "3rdParty" PROPERTY FOLDER "3rdParty")


## Set up imgui as a library --------------------------------------------------

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp")
  message(FATAL_ERROR "Dear ImGui is missing. Run git submodule update --init src/3rdParty/imgui.")
endif()

add_library(imgui
  imgui/imgui.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC imgui imgui/backends)

target_link_libraries("imgui"
  PUBLIC
    OpenGL::GL
    glfw
)

set_property(TARGET "imgui" PROPERTY FOLDER "3rdParty")
