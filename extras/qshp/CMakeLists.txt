# qshp - Westwood SHP (Tiberian Sun / Red Alert 2) support for Qt.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later.
#
# Builds against Qt 6 when it is found, Qt 5 otherwise. A plugin built with
# Qt 6.N loads in any Qt 6.M with M >= N, so build against the OLDEST Qt 6 you
# want to support.

cmake_minimum_required(VERSION 3.16)
project(qshp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Gui)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Gui)

add_library(shpcodec STATIC src/shpcodec.cpp)
target_include_directories(shpcodec PUBLIC src)
set_target_properties(shpcodec PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The plugin: one file, dropped into an application's imageformats/ folder.
add_library(qshp MODULE src/qshpplugin.cpp)
target_link_libraries(qshp PRIVATE shpcodec Qt${QT_VERSION_MAJOR}::Gui)
if(NOT WIN32)
    target_link_libraries(qshp PRIVATE ${CMAKE_DL_LIBS})
    # Keep libstdc++ private so the plugin cannot clash with the host's.
    target_link_options(qshp PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
endif()
set_target_properties(qshp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/imageformats)

# Command-line converter: SHP <-> PNG sprite sheets.
add_executable(qshp-convert src/convert.cpp)
target_link_libraries(qshp-convert PRIVATE shpcodec Qt${QT_VERSION_MAJOR}::Gui)

include(GNUInstallDirs)
install(TARGETS qshp LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/qt${QT_VERSION_MAJOR}/plugins/imageformats)
install(TARGETS qshp-convert RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
