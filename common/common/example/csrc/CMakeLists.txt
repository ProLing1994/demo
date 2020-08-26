add_executable(test_thread test_thread.cpp)
target_link_libraries(test_thread -pthread)

add_executable(demo demo.cpp)
target_link_libraries(demo)