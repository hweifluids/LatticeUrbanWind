#pragma once

#include <QString>

class vtkGenericOpenGLRenderWindow;
class vtkRenderer;

namespace Streamcenter::Index {

class IndexOpenGLCanvasBridge {
public:
    static bool validate(vtkRenderer* renderer,
                         vtkGenericOpenGLRenderWindow* renderWindow,
                         QString* errorMessage);
};

}  // namespace Streamcenter::Index
