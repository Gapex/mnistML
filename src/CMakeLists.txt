
aux_source_directory(. SRC_DIR)

add_executable(run_knn run_knn.cc knn.cc data.cc data_handler.cc)
add_executable(run_kmeans run_kmeans.cc kmeans.cc cluster.cc data.cc data_handler.cc)

target_link_libraries(run_knn pthread)
