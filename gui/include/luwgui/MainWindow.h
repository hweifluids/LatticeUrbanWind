#pragma once

#include "luwgui/BatchBoundaryPanel.h"
#include "luwgui/BoundaryCsvPanel.h"
#include "luwgui/BuildingScalePanel.h"
#include "luwgui/CommandRunner.h"
#include "luwgui/ConfigDocument.h"
#include "luwgui/ConsolePanel.h"
#include "luwgui/Preferences.h"
#include "luwgui/ProgressPanel.h"
#include "luwgui/StartupDiagnostics.h"
#include "luwgui/VtkViewWidget.h"
#include "luwgui/WavenumberPanel.h"

#include <QFileSystemModel>
#include <QHash>
#include <QList>
#include <QMainWindow>

class QComboBox;
class QAction;
class QLabel;
class QLineEdit;
class QMenu;
class QPlainTextEdit;
class QSplitter;
class QStackedWidget;
class QTabWidget;
class QToolButton;
class QTreeView;
class QTreeWidget;
class QTreeWidgetItem;
class QWidget;

namespace luwgui {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(const AppPreferences& preferences, QWidget* parent = nullptr);
    void appendStartupReport(const StartupCheckResult& startupCheck);

private:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void changeEvent(QEvent* event) override;
#ifdef Q_OS_WIN
    bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;
#endif

    struct EditorBinding {
        FieldSpec spec;
        QWidget* editor = nullptr;
        QWidget* label = nullptr;
        QString nodeId;
        bool propertyField = false;
    };

    struct TreeNodeInfo {
        QString id;
        QString title;
        QString description;
        QString parentId;
        QStringList fieldKeys;
        QStringList propertyKeys;
        bool caseRoot = false;
        bool toolsRoot = false;
        bool resultsRoot = false;
        bool managedLeaf = false;
        bool managedLeafTrashed = false;
        QString managedRole;
    };

    struct ManagedNode {
        QString id;
        QString name;
        QString title;
        QString filePath;
        QString role;
        QString storageType;
        bool trashed = false;
        int trashIndex = -1;
    };

    void buildTitleBar();
    void buildUi();
    void buildMenus();
    void applyTitleBarStyle();
    void refreshWindowButtons();
    void recordTitleBarThemeClick();
    void toggleHiddenThemeVariant();
    void configureStatusBarGeometry();
    QWidget* buildWorkflowBar();
    void rebuildSectionPages(const QString& preferredNodeId = {});
    QWidget* buildWorkflowPage();
    QWidget* buildSectionPage(const SectionSpec& section);
    QWidget* buildNodePage(const TreeNodeInfo& node);
    QWidget* buildEmptyPropertiesPage(const QString& title);
    QWidget* buildShellPlaceholderPage();
    QWidget* createPanelShell(QWidget* parent, QWidget* child, const QString& objectName);
    void registerTrackedPanel(QWidget* panelShell);
    QWidget* trackedPanelFor(QWidget* widget) const;
    void repolishTrackedPanel(QWidget* panelShell);
    void setActiveTrackedPanel(QWidget* panelShell);
    QWidget* createEditor(const FieldSpec& spec);
    QWidget* buildAuxiliaryTabs();
    void ensureCenterWorkspaceCreated();
    void ensureProjectWorkspaceLoaded();
    void showPreferencesDialog();
    void showAboutDialog();
    void applyPreferences();
    bool persistPreferences(QString* errorMessage = nullptr);
    QString preferredProjectLocation() const;
    QString preferredProjectDeckPath(RunMode mode) const;
    void syncProjectUiFromDocument();
    bool openProjectFile(const QString& path);
    void rememberRecentProjectFile(const QString& filePath);
    void trimRecentProjectFiles();
    void updateOpenRecentMenu();
    QVector<TreeNodeInfo> buildTreeNodes() const;
    QWidget* createPropertyEditor(const FieldSpec& spec);
    QString fieldDisplayLabel(const QString& nodeId, const FieldSpec& spec, bool propertyField) const;
    bool isFieldCompatible(const FieldSpec& spec) const;
    void showNavTreeContextMenu(const QPoint& pos);
    void expandNodeRecursive(QTreeWidgetItem* item, bool expand);
    void renameCase();
    void revealProjectInExplorer();
    void addManagedNode(const QString& role, const QString& type = {});
    void renameManagedNode(const QString& nodeId);
    bool commitManagedNodeRename(const QString& nodeId, const QString& nextName, QString* errorMessage = nullptr);
    void removeManagedNode(const QString& nodeId);
    void removeAllToolNodes();
    void removeAllResultNodes();
    void recoverManagedNode(const QString& nodeId);
    void permanentlyDeleteManagedNode(const QString& nodeId);
    void syncManagedStateWithProject();
    void reloadToolNodesFromGuiProperties();
    void reloadResultNodesFromGuiProperties();
    QString guiPropertiesDirectory() const;
    bool ensureGuiPropertiesDirectory();
    const ManagedNode* findToolNode(const QString& nodeId) const;
    const ManagedNode* findResultNode(const QString& nodeId) const;
    const ManagedNode* findManagedNode(const QString& nodeId) const;
    bool hasActiveDataProcessor() const;
    bool keyRequiresTreeRebuild(const QString& key) const;
    void unlockPropertyEditor(int bindingIndex);

    void newDeck(RunMode mode, const QString& caseName = {});
    QString buildSkeletonText(RunMode mode, const QString& caseName = {}) const;
    void refreshEditors();
    void refreshEditorStates();
    void refreshRawDeck();
    void refreshFileTree();
    void updateProjectAvailability();
    void setCurrentPage(const QString& pageId);
    void runPreset(CommandPreset preset);
    void runVisLuwExport();
    void runVtk2NcExport();
    void runCutVis();
    QStringList buildCutVisArguments() const;
    QString latestResultVtk() const;
    QString latestResultBaseName() const;
    QString defaultNetCdfPath() const;
    QString resolveTargetCrs() const;
    void loadLatestResult();
    void loadViewerFile(const QString& filePath);
    void fixDeckFile();
    bool isFieldActive(const QString& key) const;
    void clearEditorDisplay(const EditorBinding& binding);
    QVariant readEditorValue(const EditorBinding& binding) const;
    void writeEditorValue(const EditorBinding& binding, const QVariant& value);
    void openProject();
    bool saveProject();
    bool saveProjectAs();
    void createProjectWizard();
    void importProjectAsset(const QString& targetSubdirectory, const QString& dialogTitle);
    void stopActiveWork();
    bool ensureProjectSaved(const QString& dialogTitle);
    bool hasLoadedProject() const;
    void updateAuxiliaryPanelLayout();
    void setRightPanelVisible(bool visible);
    void syncWorkflowChromeButtons();
    void logGuiAction(const QString& message);

    ConfigDocument* document_ = nullptr;
    CommandRunner* runner_ = nullptr;
    AppPreferences preferences_;

    QWidget* titleBarWidget_ = nullptr;
    QLabel* titleBarLogo_ = nullptr;
    QLabel* titleBarTitle_ = nullptr;
    QWidget* titleMenuStrip_ = nullptr;
    QToolButton* minimizeWindowButton_ = nullptr;
    QToolButton* maximizeWindowButton_ = nullptr;
    QToolButton* closeWindowButton_ = nullptr;
    QSplitter* rootSplitter_ = nullptr;
    QSplitter* leftSplitter_ = nullptr;
    QSplitter* workspaceSplitter_ = nullptr;
    QWidget* navPanelShell_ = nullptr;
    QWidget* progressPanelShell_ = nullptr;
    QWidget* sectionPanelShell_ = nullptr;
    QTreeWidget* navTree_ = nullptr;
    ProgressPanel* progressPanel_ = nullptr;
    QStackedWidget* sectionStack_ = nullptr;
    QHash<QString, int> pageById_;
    QHash<QString, TreeNodeInfo> nodeById_;
    QHash<QString, QTreeWidgetItem*> treeItemById_;
    QVector<EditorBinding> bindings_;
    QString currentNodeId_;
    QVector<ManagedNode> toolNodes_;
    QVector<ManagedNode> resultNodes_;
    QString managedStateProjectKey_;
    bool viewTrashedTools_ = false;
    bool viewTrashedResults_ = false;

    QWidget* centerHost_ = nullptr;
    QWidget* centerPlaceholderShell_ = nullptr;
    QWidget* centerPlaceholder_ = nullptr;
    QStackedWidget* viewerStack_ = nullptr;
    QWidget* viewerPanelShell_ = nullptr;
    VtkViewWidget* vtkView_ = nullptr;
    QWidget* consolePanelShell_ = nullptr;
    ConsolePanel* console_ = nullptr;
    QSplitter* centerSplitter_ = nullptr;
    QWidget* rightHost_ = nullptr;
    QSplitter* rightSplitter_ = nullptr;
    QWidget* rightTopPanelShell_ = nullptr;
    QTabWidget* rightTopTabs_ = nullptr;
    QWidget* rightBottomPanelShell_ = nullptr;
    QTabWidget* auxiliaryTabs_ = nullptr;
    BatchBoundaryPanel* batchPanel_ = nullptr;
    BoundaryCsvPanel* boundaryCsvPanel_ = nullptr;
    WavenumberPanel* wavenumberPanel_ = nullptr;
    BuildingScalePanel* buildingScalePanel_ = nullptr;
    QList<int> consoleExpandedSizes_;
    QList<int> rightPanelExpandedSizes_;

    QPlainTextEdit* rawDeckEdit_ = nullptr;
    QFileSystemModel* fileModel_ = nullptr;
    QTreeView* fileTree_ = nullptr;
    QMenu* openRecentMenu_ = nullptr;
    QWidget* workflowBar_ = nullptr;
    QToolButton* consoleToggleButton_ = nullptr;
    QToolButton* sidePanelToggleButton_ = nullptr;
    QList<QWidget*> projectScopedWidgets_;
    QList<QAction*> projectScopedActions_;

    QLabel* workflowSummary_ = nullptr;
    QLineEdit* cutVisExtraArgsEdit_ = nullptr;
    QComboBox* modeCombo_ = nullptr;
    bool stopRequested_ = false;
    QList<qint64> titleBarThemeClickTimes_;
    QList<QWidget*> trackedPanels_;
    QWidget* activeTrackedPanel_ = nullptr;
};

}
