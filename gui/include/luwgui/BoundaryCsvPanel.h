#pragma once

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <vtkSmartPointer.h>

class QLabel;
class QVTKOpenGLNativeWidget;
class vtkActor;
class vtkGenericOpenGLRenderWindow;
class vtkLookupTable;
class vtkPolyData;
class vtkPolyDataMapper;
class vtkRenderer;
class vtkScalarBarActor;
class vtkVertexGlyphFilter;

namespace luwgui {

class BoundaryCsvPanel : public QWidget {
    Q_OBJECT

public:
    explicit BoundaryCsvPanel(QWidget* parent = nullptr);

    void setProjectDirectory(const QString& projectDirectory);

signals:
    void statusMessage(const QString& message);

private:
    void reloadLatestCsv();
    QString latestBoundaryCsv() const;
    void updateColorField();

    QString projectDirectory_;
    QString currentFile_;
    QLabel* fileLabel_ = nullptr;
    QComboBox* colorCombo_ = nullptr;
    QVTKOpenGLNativeWidget* vtkWidget_ = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkPolyData> polyData_;
    vtkSmartPointer<vtkVertexGlyphFilter> glyphFilter_;
    vtkSmartPointer<vtkPolyDataMapper> mapper_;
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkScalarBarActor> scalarBar_;
    vtkSmartPointer<vtkLookupTable> lookupTable_;
};

}
