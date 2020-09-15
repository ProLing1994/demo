add_executable(test_thread test_thread.cpp)
target_link_libraries(test_thread -pthread)

add_executable(test_reference_pointers test_reference_pointers.cpp)
target_link_libraries(test_reference_pointers)

add_executable(test_struct_size test_struct_size.cpp)
target_link_libraries(test_struct_size)

add_executable(demo demo.cpp)
target_link_libraries(demo)