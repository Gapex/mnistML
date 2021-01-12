
aux_source_directory(. SRC_DIR)

message(${SRC_DIR})

add_executable(main ${SRC_DIR})

target_link_libraries(main pthread)
