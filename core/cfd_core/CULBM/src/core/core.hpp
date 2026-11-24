#pragma once

// CULBM host foundation library (umbrella header).
//
// The implementation lives in namespace culbm::core. The solver sources call
// these helpers unqualified from the global namespace, so the names they use
// are re-declared here at global scope. Arithmetic operators of the vector
// types are found through argument-dependent lookup.
//
// Modules:
//   types.hpp          scalar aliases, constants, CONSOLE_WIDTH
//   scalar.hpp         sq/cb/sign, integer min/max, to_uint, reverse_bytes, hermite_spline
//   vec.hpp            uint3, float3, float3x3, dot/cross/length
//   text.hpp           to_string, alignment, string predicates, print_time
//   console.hpp        print/println/reprint/wait, log_info/log_warning/log_error/fatal_error
//   console_log.hpp    console lock and log file            (LUW-owned, carried over)
//   console_table.hpp  console table layout                 (LUW-owned, carried over)
//   progress.hpp       LUW GUI progress lines               (LUW-owned, carried over)
//   status_line.hpp    live status line and StatusPause
//   system.hpp         Clock, sleep, parallel_for, paths and files
//   mesh.hpp           Mesh, read_stl
//   lattice_units.hpp  Units (include it instead of this header where units are needed)

#include "types.hpp"
#include "scalar.hpp"
#include "vec.hpp"
#include "text.hpp"
#include "console.hpp"
#include "progress.hpp"
#include "status_line.hpp"
#include "system.hpp"
#include "mesh.hpp"
#include "../cuda_mpi.hpp"

// Standard headers the solver sources have always received through the host
// utility header.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using std::string;
using std::vector;
using std::thread;

using uchar = culbm::core::uchar;
using ushort = culbm::core::ushort;
using uint = culbm::core::uint;
using slong = culbm::core::slong;
using ulong = culbm::core::ulong;

using culbm::core::pi;
using culbm::core::pif;
using culbm::core::max_uint;
using culbm::core::max_ulong;
using culbm::core::CONSOLE_WIDTH;

// scalar.hpp
using culbm::core::sq;
using culbm::core::cb;
using culbm::core::sign;
using culbm::core::min;
using culbm::core::max;
using culbm::core::to_uint;
using culbm::core::reverse_bytes;
using culbm::core::hermite_spline;

// vec.hpp
using culbm::core::uint3;
using culbm::core::float3;
using culbm::core::float3x3;
using culbm::core::dot;
using culbm::core::cross;
using culbm::core::length; // float3 and std::string overloads

// text.hpp
using culbm::core::to_string;
using culbm::core::contains;
using culbm::core::begins_with;
using culbm::core::trim;
using culbm::core::substring;
using culbm::core::alignl;
using culbm::core::alignr;
using culbm::core::alignc;
using culbm::core::print_time;

// console.hpp, console_log.hpp, progress.hpp, status_line.hpp
using culbm::core::print;
using culbm::core::println;
using culbm::core::reprint;
using culbm::core::wait;
using culbm::core::log_info;
using culbm::core::log_warning;
using culbm::core::log_error;
using culbm::core::fatal_error;
using culbm::core::set_console_log_file;
using culbm::core::close_console_log_file;
using culbm::core::get_console_log_file;
using culbm::core::luw_progress_gui_mode;
using culbm::core::luw_emit_progress;
using culbm::core::StatusPause;

// system.hpp
using culbm::core::Clock;
using culbm::core::sleep;
using culbm::core::parallel_for;
using culbm::core::get_exe_path;
using culbm::core::create_folder;
using culbm::core::create_file_extension;
using culbm::core::write_file;

// mesh.hpp (Mesh itself is a global type)
using culbm::core::read_stl;
