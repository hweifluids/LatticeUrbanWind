#pragma once

#include "luwgui/BatchBoundaryPanel.h"
#include "luwgui/BoundaryCsvPanel.h"
#include "luwgui/BuildingScalePanel.h"
#include "luwgui/CommandRunner.h"
#include "luwgui/ConfigDocument.h"
#include "luwgui/ConsolePanel.h"
#include "luwgui/Preferences.h"
#include "luwgui/VtkViewWidget.h"
#include "luwgui/WavenumberPanel.h"

#include <QFileSystemModel>
#include <QHash>
#include <QList>
#include <QMainWindow>

class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QSplitter;
class QStackedWidget;
class QTabWidget;
class QTreeView;
class QTreeWidget;
class QWidget;

namespace luwgui {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(const AppPreferences& preferences, QWidget* parent = nullptr);

private:
    struct EditorBinding {
        FieldSpec spec;
        QWidget* editor = nullptr;
    };

    void buildUi();
    void buildMenus();
    void rebuildSectionPages();
    QWidget* buildWorkflowPage();
    QWidget* buildSectionPage(const SectionSpec& section);
    QWidget* createEditor(const FieldSpec& spec);
    QWidget* buildAuxiliaryTabs();
    void showPreferencesDialog();
    void showAboutDialog();
    void applyPreferences();

    void newDeck(RunMode mode, const QString& caseName = {});
    QString buildSkeletonText(RunMode mode, const QString& caseName = {}) const;
    void refreshEditors();
    void refreshRawDeck();
    void refreshFileTree();
    void setCurrentPage(const QString& pageId);
    void runPreset(CommandPreset preset);
    void runCutVis();
    QStringList buildCutVisArguments() const;
    QString latestResultVtk() const;
    void loadLatestResult();
    void loadViewerFile(const QString& filePath);
    QVariant readEditorValue(const EditorBinding& binding) const;
    void writeEditorValue(const EditorBinding& binding, const QVariant& value);
    void openProject();
    bool saveProject();
    bool saveProjectAs();
    void createProjectWizard();
    void importProjectAsset(const QString& targetSubdirectory, const QString& dialogTitle);
    void stopActiveWork();
    void updateAuxiliaryPanelLayout();
    void logGuiAction(const QString& message);

    ConfigDocument* document_ = nullptr;
    CommandRunner* runner_ = nullptr;
    AppPreferences preferences_;

    QTreeWidget* navTree_ = nullptr;
    QStackedWidget* sectionStack_ = nullptr;
    QHash<QString, int> pageById_;
    QVector<EditorBinding> bindings_;

    VtkViewWidget* vtkView_ = nullptr;
    ConsolePanel* console_ = nullptr;
    QSplitter* centerSplitter_ = nullptr;
    QSplitter* rightSplitter_ = nullptr;
    QTabWidget* rightTopTabs_ = nullptr;
    BatchBoundaryPanel* batchPanel_ = nullptr;
    BoundaryCsvPanel* boundaryCsvPanel_ = nullptr;
    WavenumberPanel* wavenumberPanel_ = nullptr;
    BuildingScalePanel* buildingScalePanel_ = nullptr;
    QList<int> consoleExpandedSizes_;

    QPlainTextEdit* rawDeckEdit_ = nullptr;
    QFileSystemModel* fileModel_ = nullptr;
    QTreeView* fileTree_ = nullptr;

    QLabel* workflowSummary_ = nullptr;
    QLineEdit* cutVisExtraArgsEdit_ = nullptr;
    QComboBox* modeCombo_ = nullptr;
};

}
