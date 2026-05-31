#include "index/IndexOpenGLCanvas.h"

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>

namespace Streamcenter::Index {

bool IndexOpenGLCanvasBridge::validate(vtkRenderer* renderer,
                                       vtkGenericOpenGLRenderWindow* renderWindow,
                                       QString* errorMessage) {
    if (renderer == nullptr || renderWindow == nullptr) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("NVIDIA IndeX render bridge requires a live VTK OpenGL renderer.");
        }
        return false;
    }
    const int* size = renderWindow->GetSize();
    if (size == nullptr || size[0] <= 0 || size[1] <= 0) {
        if (errorMessage != nullptr) {
            *errorMessage = QStringLiteral("NVIDIA IndeX render bridge requires a non-empty OpenGL framebuffer.");
        }
        return false;
    }
    return true;
}

}  // namespace Streamcenter::Index
