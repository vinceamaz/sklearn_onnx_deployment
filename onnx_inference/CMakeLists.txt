cmake_minimum_required (VERSION 3.8)

add_executable (${PROJECT_NAME} "main.cpp" "main.h" )

target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

# 将目录拷贝到 .exe 目录

# Specify the directories you want to copy
set(directories_to_copy
    "${CMAKE_SOURCE_DIR}/input"
    "${CMAKE_SOURCE_DIR}/model"
)

# Copy each directory and its files using copy_directory command
foreach(directory ${directories_to_copy})
    get_filename_component(directory_name ${directory} NAME)
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${directory} 
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${directory_name}
    )
endforeach()

foreach(directory ${directories_to_copy})
    file(GLOB_RECURSE files_to_copy ${directory}/*)
    foreach(file ${files_to_copy})
        file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${file})
        get_filename_component(destination_dir ${relative_path} DIRECTORY)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/${destination_dir}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${file}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/${relative_path}
        )
    endforeach()
endforeach()