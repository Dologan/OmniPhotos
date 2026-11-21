set(MODULE_NAME Test)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")


find_package(GTest 1.14 CONFIG QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723 # v1.17.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(${MODULE_NAME}
  ${sources}
  ${headers}
)

target_include_directories(${MODULE_NAME}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${MODULE_NAME}
  3rdParty
  Core
  Utils

  PreprocessingApp

  ${OpenCV_LIBS}
  glog::glog
  GTest::gtest
  OpenGL::GL
)

set_target_properties(${MODULE_NAME} PROPERTIES FOLDER Tests)
set(OMNIPHOTOS_TEST_DATA "" CACHE PATH "Directory containing the OmniPhotos C++ test fixtures")
if(OMNIPHOTOS_TEST_DATA)
  add_test(NAME OmniPhotos.Native COMMAND Test -f "${OMNIPHOTOS_TEST_DATA}/")
endif()
