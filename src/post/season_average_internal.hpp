#pragma once

#include <filesystem>
#include <iosfwd>

namespace luw::post::detail {

void renderSeasonAverageFigures(const std::filesystem::path& vtkPath,
                                const std::filesystem::path& figureDirectory,
                                int dpi,
                                std::ostream& output);

} // namespace luw::post::detail
