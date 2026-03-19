#include "luwgui/MainWindow.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QFont>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSizePolicy>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QStyle>
#include <QTabWidget>
#include <QTreeView>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <set>

namespace luwgui {

namespace {

QVariantList parseLooseList(const QString& text, bool integerMode) {
    QVariantList out;
    QString normalized = text;
    normalized.replace('[', ' ');
    normalized.replace(']', ' ');
    const QStringList parts = normalized.split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
    for (const QString& part : parts) {
        bool ok = false;
        if (integerMode) {
            const int value = part.toInt(&ok);
            if (ok) {
                out.push_back(value);
            }
        } else {
            const double value = part.toDouble(&ok);
            if (ok) {
                out.push_back(value);
            }
        }
    }
    return out;
}

QString renderVariantList(const QVariantList& list, bool integerMode) {
    QStringList parts;
    for (const QVariant& value : list) {
        parts.push_back(integerMode ? QString::number(value.toInt()) : QString::number(value.toDouble(), 'f', 6));
    }
    return parts.join(",   ");
}

bool copyFileReplacing(const QString& sourcePath, const QString& targetPath, QString* errorMessage = nullptr) {
    if (QFileInfo(sourcePath).absoluteFilePath() == QFileInfo(targetPath).absoluteFilePath()) {
        return true;
    }
    QFile::remove(targetPath);
    if (QFile::copy(sourcePath, targetPath)) {
        return true;
    }
    if (errorMessage) {
        *errorMessage = QString("Failed to copy %1 to %2.").arg(sourcePath, targetPath);
    }
    return false;
}

QStringList companionExtensionsFor(const QString& suffix) {
    const QString ext = suffix.toLower();
    if (ext == "shp") {
        return {"shp", "shx", "dbf", "prj", "cpg", "qix", "sbn", "sbx"};
    }
    if (ext == "tif" || ext == "tiff") {
        return {"tif", "tiff", "tfw", "prj", "aux.xml", "ovr"};
    }
    return {suffix};
}

} // namespace

MainWindow::MainWindow(const AppPreferences& preferences, QWidget* parent)
    : QMainWindow(parent)
    , document_(new ConfigDocument(this))
    , runner_(new CommandRunner(this))
    , preferences_(preferences) {
    runner_->setDocument(document_);
    buildUi();
    buildMenus();

    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshEditors);
    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshRawDeck);
    connect(document_, &ConfigDocument::changed, this, &MainWindow::refreshFileTree);
    connect(document_, &ConfigDocument::changed, this, [this] {
        if (vtkView_) {
            vtkView_->setProjectDirectory(document_->projectDirectory());
        }
        if (boundaryCsvPanel_) {
            boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
        }
    });
    connect(document_, &ConfigDocument::documentReloaded, this, [this] {
        if (vtkView_) {
            vtkView_->setProjectDirectory(document_->projectDirectory());
        }
        if (boundaryCsvPanel_) {
            boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
        }
    });
    connect(document_, &ConfigDocument::externalFileReloaded, this, [this](const QString& filePath) {
        const QString message = "Reloaded backend-updated deck: " + QDir::toNativeSeparators(filePath);
        statusBar()->showMessage(message, 5000);
        console_->appendText("[GUI] " + message + "\n");
    });
    connect(document_, &ConfigDocument::externalFileReloadFailed, this, [this](const QString& filePath, const QString& error) {
        const QString message = QString("Failed to reload updated deck %1: %2")
            .arg(QDir::toNativeSeparators(filePath), error);
        statusBar()->showMessage(message, 5000);
        console_->appendText("[GUI] " + message + "\n");
    });
    connect(document_, &ConfigDocument::modeChanged, this, [this](RunMode mode) {
        if (modeCombo_) {
            modeCombo_->setCurrentIndex(static_cast<int>(mode));
        }
        rebuildSectionPages();
        updateAuxiliaryPanelLayout();
        refreshEditors();
    });
    connect(runner_, &CommandRunner::outputReady, console_, &ConsolePanel::appendText);
    connect(runner_, &CommandRunner::errorText, console_, &ConsolePanel::appendText);
    connect(runner_, &CommandRunner::started, this, [this](const QString& title) {
        statusBar()->showMessage("Running: " + title);
        console_->appendText("\n[GUI] Running " + title + "\n");
    });
    connect(runner_, &CommandRunner::finished, this, [this](const QString& title, int exitCode, QProcess::ExitStatus) {
        statusBar()->showMessage(title + " finished with exit code " + QString::number(exitCode), 5000);
        console_->appendText("[GUI] " + title + " finished with exit code " + QString::number(exitCode) + "\n");
        QString reloadError;
        if (!document_->reloadFromDisk(&reloadError) && !document_->filePath().isEmpty() && !reloadError.trimmed().isEmpty()) {
            console_->appendText("[GUI] Deck reload skipped: " + reloadError + "\n");
        }
        if (title == "Solve" && exitCode == 0 && vtkView_) {
            vtkView_->handleSolverFinished();
        }
    });
    connect(vtkView_, &VtkViewWidget::statusMessage, this, [this](const QString& message) {
        statusBar()->showMessage(message, 5000);
    });
    connect(vtkView_, &VtkViewWidget::guiActionRequested, this, &MainWindow::logGuiAction);
    connect(vtkView_, &VtkViewWidget::fileLoaded, this, [this](const QString& filePath) {
        if (filePath.endsWith(".vtk", Qt::CaseInsensitive) && wavenumberPanel_) {
            const bool autoAnalyze = QRegularExpression(R"(_raw_u-\d+\.vtk$)", QRegularExpression::CaseInsensitiveOption)
                .match(QFileInfo(filePath).fileName())
                .hasMatch();
            wavenumberPanel_->setSuggestedFilePath(filePath, autoAnalyze);
        }
    });
    connect(console_, &ConsolePanel::collapseToggled, this, [this](bool collapsed) {
        if (!centerSplitter_) {
            return;
        }
        if (collapsed) {
            consoleExpandedSizes_ = centerSplitter_->sizes();
            const QList<int> sizes = consoleExpandedSizes_.isEmpty() ? QList<int>{1, 1} : consoleExpandedSizes_;
            const int total = sizes.value(0, 1) + sizes.value(1, 1);
            centerSplitter_->setSizes({std::max(total - console_->collapsedHeight(), 1), console_->collapsedHeight()});
        } else if (consoleExpandedSizes_.size() == 2) {
            centerSplitter_->setSizes(consoleExpandedSizes_);
        } else {
            centerSplitter_->setSizes({760, 220});
        }
    });
    connect(wavenumberPanel_, &WavenumberPanel::statusMessage, this, [this](const QString& message) {
        statusBar()->showMessage(message, 5000);
    });
    connect(wavenumberPanel_, &WavenumberPanel::guiActionRequested, this, &MainWindow::logGuiAction);
    connect(buildingScalePanel_, &BuildingScalePanel::statusMessage, this, [this](const QString& message) {
        statusBar()->showMessage(message, 5000);
    });
    connect(boundaryCsvPanel_, &BoundaryCsvPanel::statusMessage, this, [this](const QString& message) {
        statusBar()->showMessage(message, 5000);
    });

    newDeck(RunMode::Luw);
    vtkView_->setProjectDirectory(document_->projectDirectory());
    boundaryCsvPanel_->setProjectDirectory(document_->projectDirectory());
    updateAuxiliaryPanelLayout();
    applyPreferences();
    resize(1800, 1040);
}

void MainWindow::buildUi() {
    auto* root = new QSplitter(Qt::Horizontal, this);
    setCentralWidget(root);

    auto* leftSplitter = new QSplitter(Qt::Vertical, root);
    auto* center = new QSplitter(Qt::Vertical, root);
    centerSplitter_ = center;
    auto* rightSplitter = new QSplitter(Qt::Vertical, root);
    rightSplitter_ = rightSplitter;
    root->addWidget(leftSplitter);
    root->addWidget(center);
    root->addWidget(rightSplitter);
    root->setStretchFactor(0, 18);
    root->setStretchFactor(1, 58);
    root->setStretchFactor(2, 24);

    navTree_ = new QTreeWidget(leftSplitter);
    navTree_->setHeaderHidden(true);
    leftSplitter->addWidget(navTree_);

    sectionStack_ = new QStackedWidget(leftSplitter);
    leftSplitter->addWidget(sectionStack_);
    leftSplitter->setStretchFactor(0, 3);
    leftSplitter->setStretchFactor(1, 5);

    vtkView_ = new VtkViewWidget(center);
    console_ = new ConsolePanel(center);
    center->addWidget(vtkView_);
    center->addWidget(console_);
    center->setStretchFactor(0, 3);
    center->setStretchFactor(1, 1);

    rightTopTabs_ = new QTabWidget(rightSplitter);
    batchPanel_ = new BatchBoundaryPanel(rightTopTabs_);
    boundaryCsvPanel_ = new BoundaryCsvPanel(rightTopTabs_);
    rightTopTabs_->addTab(batchPanel_, "Batch Boundary");
    rightTopTabs_->addTab(boundaryCsvPanel_, "Boundary CSV");
    rightSplitter->addWidget(rightTopTabs_);
    rightSplitter->addWidget(buildAuxiliaryTabs());
    rightSplitter->setStretchFactor(0, 3);
    rightSplitter->setStretchFactor(1, 2);

    batchPanel_->setDocument(document_);

    connect(navTree_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem* current) {
        if (!current) {
            return;
        }
        setCurrentPage(current->data(0, Qt::UserRole).toString());
    });

    statusBar()->showMessage("Ready");
    consoleExpandedSizes_ = {760, 220};
    rebuildSectionPages();
}

void MainWindow::buildMenus() {
    auto* fileMenu = menuBar()->addMenu("File");
    fileMenu->addAction("New Project", this, &MainWindow::createProjectWizard);
    fileMenu->addAction("Open Project", this, &MainWindow::openProject);
    fileMenu->addSeparator();
    fileMenu->addAction("Save", this, [this] { saveProject(); });
    fileMenu->addAction("Save As", this, [this] { saveProjectAs(); });
    fileMenu->addSeparator();
    auto* importMenu = fileMenu->addMenu("Import");
    importMenu->addAction("Terrain Database", this, [this] {
        importProjectAsset("terrain_db", "Import Terrain Database");
    });
    importMenu->addAction("Building Database", this, [this] {
        importProjectAsset("building_db", "Import Building Database");
    });
    importMenu->addAction("Wind Database", this, [this] {
        importProjectAsset("wind_bc", "Import Wind Database");
    });

    auto* runMenu = menuBar()->addMenu("Run");
    runMenu->addAction("Run Full Workflow", this, [this] { runPreset(CommandPreset::FullWorkflow); });
    runMenu->addAction("Build Batch Geometry", this, [this] { runPreset(CommandPreset::PrepareBatchGeometry); });
    runMenu->addAction("Solve", this, [this] { runPreset(CommandPreset::Solve); });
    runMenu->addAction("Stop", this, &MainWindow::stopActiveWork);
    runMenu->addSeparator();
    runMenu->addAction("Generate Visualization Data", this, [this] { runPreset(CommandPreset::VisLuw); });
    runMenu->addAction("Export NetCDF", this, [this] { runPreset(CommandPreset::Vtk2Nc); });
    runMenu->addAction("Generate Cut Visuals", this, &MainWindow::runCutVis);

    auto* viewMenu = menuBar()->addMenu("View");
    viewMenu->addAction("Load Latest Result", this, &MainWindow::loadLatestResult);
    viewMenu->addAction("Reset Camera", vtkView_, &VtkViewWidget::resetCamera);
    viewMenu->addAction("Save View Image", vtkView_, &VtkViewWidget::saveImage);

    auto* helpMenu = menuBar()->addMenu("About");
    helpMenu->addAction("Preferences", this, &MainWindow::showPreferencesDialog);
    helpMenu->addAction("About LUW Studio", this, &MainWindow::showAboutDialog);
}

void MainWindow::rebuildSectionPages() {
    bindings_.clear();
    pageById_.clear();
    while (sectionStack_->count() > 0) {
        QWidget* widget = sectionStack_->widget(0);
        sectionStack_->removeWidget(widget);
        widget->deleteLater();
    }
    navTree_->clear();

    auto addPage = [&](const QString& id, const QString& title, QWidget* page) {
        pageById_.insert(id, sectionStack_->addWidget(page));
        auto* item = new QTreeWidgetItem(QStringList{title});
        item->setData(0, Qt::UserRole, id);
        navTree_->addTopLevelItem(item);
    };

    addPage("workflow", "Workflow", buildWorkflowPage());
    for (const SectionSpec& section : sectionSpecs()) {
        addPage(section.id, section.title, buildSectionPage(section));
    }
    navTree_->setCurrentItem(navTree_->topLevelItem(0));
}

QWidget* MainWindow::buildWorkflowPage() {
    auto* page = new QWidget(sectionStack_);
    auto* root = new QVBoxLayout(page);
    root->setContentsMargins(8, 8, 8, 8);

    auto* summaryBox = new QGroupBox("Project Summary", page);
    auto* summaryLayout = new QVBoxLayout(summaryBox);
    auto* summaryControls = new QWidget(summaryBox);
    auto* summaryControlsLayout = new QHBoxLayout(summaryControls);
    summaryControlsLayout->setContentsMargins(0, 0, 0, 0);
    summaryControlsLayout->addWidget(new QLabel("Mode", summaryControls));
    modeCombo_ = new QComboBox(summaryControls);
    modeCombo_->addItems({"LUW", "LUWDG", "LUWPF"});
    modeCombo_->setCurrentIndex(static_cast<int>(document_->mode()));
    summaryControlsLayout->addWidget(modeCombo_);
    summaryControlsLayout->addStretch(1);
    summaryLayout->addWidget(summaryControls);
    workflowSummary_ = new QLabel(summaryBox);
    workflowSummary_->setWordWrap(true);
    summaryLayout->addWidget(workflowSummary_);
    root->addWidget(summaryBox);

    auto* workflowBox = new QGroupBox("Workflow Actions", page);
    auto* workflowLayout = new QGridLayout(workflowBox);
    auto* runWorkflowButton = new QPushButton("Run Full Workflow", workflowBox);
    auto* cdfButton = new QPushButton("Inspect Wind Climate Inputs", workflowBox);
    auto* shpButton = new QPushButton("Inspect Building Footprints", workflowBox);
    auto* bcButton = new QPushButton("Generate Boundary Conditions", workflowBox);
    auto* cutButton = new QPushButton("Crop Geometry Domain", workflowBox);
    auto* voxButton = new QPushButton("Generate Voxel Domain", workflowBox);
    auto* valButton = new QPushButton("Validate Case Setup", workflowBox);
    auto* batchButton = new QPushButton("Build Batch Geometry", workflowBox);
    auto* solveButton = new QPushButton("Solve", workflowBox);
    auto* stopButton = new QPushButton("Stop", workflowBox);
    workflowLayout->addWidget(runWorkflowButton, 0, 0, 1, 2);
    workflowLayout->addWidget(cdfButton, 1, 0);
    workflowLayout->addWidget(shpButton, 1, 1);
    workflowLayout->addWidget(bcButton, 2, 0);
    workflowLayout->addWidget(cutButton, 2, 1);
    workflowLayout->addWidget(voxButton, 3, 0);
    workflowLayout->addWidget(valButton, 3, 1);
    workflowLayout->addWidget(batchButton, 4, 0);
    workflowLayout->addWidget(solveButton, 4, 1);
    workflowLayout->addWidget(stopButton, 5, 0, 1, 2);
    root->addWidget(workflowBox);

    auto* postBox = new QGroupBox("Post Processing", page);
    auto* postLayout = new QGridLayout(postBox);
    auto* visButton = new QPushButton("Generate Visualization Data", postBox);
    auto* ncButton = new QPushButton("Export NetCDF", postBox);
    auto* latestButton = new QPushButton("Load Latest Result", postBox);
    auto* cutVisButton = new QPushButton("Generate Cut Visuals", postBox);
    cutVisExtraArgsEdit_ = new QLineEdit(postBox);
    cutVisExtraArgsEdit_->setPlaceholderText("--export-cropped-vtk --cell-size 10");
    postLayout->addWidget(visButton, 0, 0);
    postLayout->addWidget(ncButton, 0, 1);
    postLayout->addWidget(latestButton, 1, 0);
    postLayout->addWidget(cutVisButton, 1, 1);
    postLayout->addWidget(new QLabel("Cut visual extra arguments"), 2, 0, 1, 2);
    postLayout->addWidget(cutVisExtraArgsEdit_, 3, 0, 1, 2);
    root->addWidget(postBox);
    root->addStretch(1);

    connect(runWorkflowButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::FullWorkflow); });
    connect(cdfButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::CdfInspect); });
    connect(shpButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::ShpInspect); });
    connect(bcButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::BuildBoundaryConditions); });
    connect(cutButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::CutGeometry); });
    connect(voxButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Voxelize); });
    connect(valButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Validate); });
    connect(batchButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::PrepareBatchGeometry); });
    connect(solveButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Solve); });
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopActiveWork);
    connect(visButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::VisLuw); });
    connect(ncButton, &QPushButton::clicked, this, [this] { runPreset(CommandPreset::Vtk2Nc); });
    connect(latestButton, &QPushButton::clicked, this, &MainWindow::loadLatestResult);
    connect(cutVisButton, &QPushButton::clicked, this, &MainWindow::runCutVis);
    connect(modeCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        const RunMode mode = static_cast<RunMode>(std::clamp(index, 0, 2));
        document_->setMode(mode);
    });

    return page;
}

QWidget* MainWindow::buildSectionPage(const SectionSpec& section) {
    auto* page = new QWidget(sectionStack_);
    auto* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);

    auto* scroll = new QScrollArea(page);
    scroll->setWidgetResizable(true);
    pageLayout->addWidget(scroll);

    auto* container = new QWidget(scroll);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(8, 8, 8, 8);
    auto* description = new QLabel(section.description, container);
    description->setWordWrap(true);
    description->setProperty("muted", true);
    layout->addWidget(description);

    auto* formContainer = new QWidget(container);
    auto* form = new QFormLayout(formContainer);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignTop);
    form->setFormAlignment(Qt::AlignTop);
    form->setContentsMargins(0, 0, 0, 0);

    for (const FieldSpec& spec : fieldsForSection(section.id, document_->mode())) {
        QWidget* editor = createEditor(spec);
        editor->setEnabled(!spec.readOnly);
        editor->setToolTip(spec.help);
        bindings_.push_back({spec, editor});
        form->addRow(spec.label, editor);
    }

    layout->addWidget(formContainer);
    layout->addStretch(1);
    scroll->setWidget(container);
    return page;
}

QWidget* MainWindow::createEditor(const FieldSpec& spec) {
    QWidget* editor = nullptr;

    switch (spec.kind) {
    case FieldKind::Boolean: {
        auto* box = new QCheckBox();
        connect(box, &QCheckBox::toggled, this, [this, spec](bool value) {
            document_->setTypedValue(spec.key, value);
        });
        editor = box;
        break;
    }
    case FieldKind::Enum: {
        auto* combo = new QComboBox();
        combo->addItems(spec.enumValues);
        connect(combo, &QComboBox::currentTextChanged, this, [this, spec](const QString& text) {
            document_->setTypedValue(spec.key, text);
        });
        editor = combo;
        break;
    }
    case FieldKind::Integer: {
        auto* spin = new QSpinBox();
        spin->setRange(-999999999, 999999999);
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, [this, spec](int value) {
            document_->setTypedValue(spec.key, value);
        });
        editor = spin;
        break;
    }
    case FieldKind::Float: {
        auto* spin = new QDoubleSpinBox();
        spin->setRange(-1.0e9, 1.0e9);
        spin->setDecimals(6);
        spin->setSingleStep(0.1);
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, spec](double value) {
            document_->setTypedValue(spec.key, value);
        });
        editor = spin;
        break;
    }
    case FieldKind::Multiline: {
        auto* text = new QPlainTextEdit();
        text->setMinimumHeight(110);
        connect(text, &QPlainTextEdit::textChanged, this, [this, spec, text] {
            document_->setTypedValue(spec.key, text->toPlainText());
        });
        editor = text;
        break;
    }
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::UIntTriplet:
    case FieldKind::FloatList:
    case FieldKind::TokenList:
    case FieldKind::String: {
        auto* line = new QLineEdit();
        connect(line, &QLineEdit::editingFinished, this, [this, spec, line] {
            document_->setTypedValue(spec.key, readEditorValue({spec, line}));
        });
        editor = line;
        break;
    }
    }

    return editor;
}

QWidget* MainWindow::buildAuxiliaryTabs() {
    auto* tabs = new QTabWidget(this);

    wavenumberPanel_ = new WavenumberPanel(tabs);
    tabs->addTab(wavenumberPanel_, "Wavenumber");

    buildingScalePanel_ = new BuildingScalePanel(tabs);
    tabs->addTab(buildingScalePanel_, "Building Scale");

    auto* filesPage = new QWidget(tabs);
    auto* filesLayout = new QVBoxLayout(filesPage);
    fileModel_ = new QFileSystemModel(filesPage);
    fileModel_->setRootPath(document_->projectDirectory());
    fileTree_ = new QTreeView(filesPage);
    fileTree_->setModel(fileModel_);
    fileTree_->setRootIndex(fileModel_->index(document_->projectDirectory()));
    fileTree_->setAlternatingRowColors(true);
    fileTree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    filesLayout->addWidget(fileTree_);
    tabs->addTab(filesPage, "Files");

    auto* rawPage = new QWidget(tabs);
    auto* rawLayout = new QVBoxLayout(rawPage);
    rawDeckEdit_ = new QPlainTextEdit(rawPage);
    rawLayout->addWidget(rawDeckEdit_, 1);
    auto* rawButtons = new QWidget(rawPage);
    auto* rawButtonsLayout = new QHBoxLayout(rawButtons);
    rawButtonsLayout->setContentsMargins(0, 0, 0, 0);
    auto* applyButton = new QPushButton("Apply Raw Deck", rawButtons);
    auto* saveButton = new QPushButton("Save Deck", rawButtons);
    rawButtonsLayout->addWidget(applyButton);
    rawButtonsLayout->addWidget(saveButton);
    rawButtonsLayout->addStretch(1);
    rawLayout->addWidget(rawButtons);
    tabs->addTab(rawPage, "Raw Deck");

    connect(applyButton, &QPushButton::clicked, this, [this] {
        QString error;
        if (!document_->applyRawText(rawDeckEdit_->toPlainText(), &error)) {
            QMessageBox::critical(this, "Apply raw deck", error);
            return;
        }
        logGuiAction("Applied raw deck edits");
    });
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveProject);
    connect(fileTree_, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        const QString path = fileModel_->filePath(index);
        if (path.endsWith(".vtk", Qt::CaseInsensitive) || path.endsWith(".stl", Qt::CaseInsensitive)) {
            loadViewerFile(path);
            if (path.endsWith(".vtk", Qt::CaseInsensitive)) {
                wavenumberPanel_->setSuggestedFilePath(path);
            }
        } else if (path.endsWith(".shp", Qt::CaseInsensitive)) {
            buildingScalePanel_->setSuggestedFilePath(path);
        }
    });

    return tabs;
}

void MainWindow::showPreferencesDialog() {
    QDialog dialog(this);
    dialog.setWindowTitle("Preferences");
    dialog.resize(420, 180);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    auto* themeCombo = new QComboBox(&dialog);
    themeCombo->addItems({"Dark", "Light"});
    themeCombo->setCurrentText(themeModeDisplayName(preferences_.themeMode));
    form->addRow("Appearance", themeCombo);
    form->addRow("Preference file", new QLabel(QDir::toNativeSeparators(preferencesFilePath()), &dialog));
    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    preferences_.themeMode = themeModeFromString(themeCombo->currentText());
    QString error;
    if (!savePreferences(preferences_, &error)) {
        QMessageBox::critical(this, "Preferences", error);
        return;
    }
    applyPreferences();
    statusBar()->showMessage("Preferences saved.", 4000);
}

void MainWindow::showAboutDialog() {
    QDialog dialog(this);
    dialog.setWindowTitle("About LUW Studio");
    dialog.resize(520, 280);

    auto* layout = new QVBoxLayout(&dialog);
    auto* title = new QLabel("LUW Studio", &dialog);
    QFont titleFont = title->font();
    titleFont.setBold(true);
    titleFont.setPointSize(titleFont.pointSize() + 4);
    title->setFont(titleFont);
    layout->addWidget(title);

    auto* body = new QLabel(
        "Engineering GUI for LatticeUrbanWind.\n\n"
        "Features in this build include structured deck editing, workflow orchestration, embedded VTK post-processing, "
        "mode-aware LUW/LUWDG/LUWPF controls, console forwarding, cut_vis launch support, wavenumber spectrum analysis, "
        "and weighted building-scale distribution analysis.\n\n"
        "Preferences are stored in gui/preferences.json and loaded automatically at startup.",
        &dialog);
    body->setWordWrap(true);
    layout->addWidget(body, 1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    layout->addWidget(buttons);

    dialog.exec();
}

void MainWindow::applyPreferences() {
    applyTheme(*qApp, preferences_.themeMode);
}

void MainWindow::openProject() {
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Open Project",
        document_->projectDirectory(),
        "LUW Deck (*.luw *.luwdg *.luwpf)");
    if (path.isEmpty()) {
        return;
    }
    QString error;
    if (!document_->loadFromFile(path, &error)) {
        QMessageBox::critical(this, "Open project", error);
        return;
    }
    logGuiAction("Opened project " + QFileInfo(path).fileName());
}

bool MainWindow::saveProject() {
    QString error;
    if (document_->filePath().isEmpty()) {
        return saveProjectAs();
    }
    if (!document_->save(&error)) {
        QMessageBox::critical(this, "Save project", error);
        return false;
    }
    logGuiAction("Saved project " + QFileInfo(document_->filePath()).fileName());
    return true;
}

bool MainWindow::saveProjectAs() {
    const QString path = QFileDialog::getSaveFileName(
        this,
        "Save Project As",
        QDir(document_->projectDirectory()).filePath(defaultDeckName(document_->mode())),
        "LUW Deck (*.luw *.luwdg *.luwpf)");
    if (path.isEmpty()) {
        return false;
    }
    QString error;
    if (!document_->saveAs(path, &error)) {
        QMessageBox::critical(this, "Save project", error);
        return false;
    }
    logGuiAction("Saved project as " + QFileInfo(path).fileName());
    return true;
}

void MainWindow::createProjectWizard() {
    QDialog dialog(this);
    dialog.setWindowTitle("New Project");
    dialog.resize(520, 220);

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();
    auto* folderRow = new QWidget(&dialog);
    auto* folderLayout = new QHBoxLayout(folderRow);
    folderLayout->setContentsMargins(0, 0, 0, 0);
    auto* folderEdit = new QLineEdit(folderRow);
    folderEdit->setText(document_->projectDirectory());
    auto* browseButton = new QPushButton("Browse", folderRow);
    folderLayout->addWidget(folderEdit, 1);
    folderLayout->addWidget(browseButton);
    form->addRow("Parent folder", folderRow);

    auto* caseEdit = new QLineEdit(&dialog);
    caseEdit->setText(document_->hasKey("casename") ? document_->typedValue("casename").toString() : "case_luw");
    form->addRow("Case name", caseEdit);

    auto* modeCombo = new QComboBox(&dialog);
    modeCombo->addItems({"LUW", "LUWDG", "LUWPF"});
    modeCombo->setCurrentIndex(static_cast<int>(document_->mode()));
    form->addRow("Run mode", modeCombo);
    layout->addLayout(form);
    layout->addStretch(1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);

    connect(browseButton, &QPushButton::clicked, &dialog, [&dialog, folderEdit] {
        const QString path = QFileDialog::getExistingDirectory(&dialog, "Select Parent Folder", folderEdit->text());
        if (!path.isEmpty()) {
            folderEdit->setText(path);
        }
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    const QString parentFolder = folderEdit->text().trimmed();
    const QString caseName = caseEdit->text().trimmed();
    if (parentFolder.isEmpty()) {
        QMessageBox::warning(this, "New project", "Select a parent folder.");
        return;
    }
    if (caseName.isEmpty()) {
        QMessageBox::warning(this, "New project", "Enter a case name.");
        return;
    }
    if (caseName.contains(QRegularExpression(R"([\\/:*?"<>|])"))) {
        QMessageBox::warning(this, "New project", "Case name contains invalid path characters.");
        return;
    }

    const RunMode mode = static_cast<RunMode>(std::clamp(modeCombo->currentIndex(), 0, 2));
    const QString caseDirectory = QDir(parentFolder).filePath(caseName);
    const QDir caseDir(caseDirectory);
    if (caseDir.exists()) {
        const QFileInfoList existingEntries = caseDir.entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries);
        if (!existingEntries.isEmpty()) {
            const auto answer = QMessageBox::question(
                this,
                "New project",
                "The case folder already exists and is not empty. Continue and overwrite the project file?");
            if (answer != QMessageBox::Yes) {
                return;
            }
        }
    }

    for (const QString& subdirectory : {"building_db", "proj_temp", "RESULTS", "terrain_db", "wind_bc"}) {
        if (!QDir().mkpath(QDir(caseDirectory).filePath(subdirectory))) {
            QMessageBox::critical(this, "New project", "Failed to create " + subdirectory + ".");
            return;
        }
    }

    newDeck(mode, caseName);

    QString error;
    const QString deckPath = QDir(caseDirectory).filePath(defaultDeckName(mode));
    if (!document_->saveAs(deckPath, &error)) {
        QMessageBox::critical(this, "New project", error);
        return;
    }

    logGuiAction("Created project " + caseName + " in " + QDir::toNativeSeparators(caseDirectory));
}

void MainWindow::importProjectAsset(const QString& targetSubdirectory, const QString& dialogTitle) {
    if (document_->filePath().isEmpty()) {
        QMessageBox::warning(this, "Import", "Create or open a project before importing files.");
        return;
    }

    const QStringList selectedFiles = QFileDialog::getOpenFileNames(this, dialogTitle, document_->projectDirectory(), "All files (*.*)");
    if (selectedFiles.isEmpty()) {
        return;
    }

    const QString targetDirectory = QDir(document_->projectDirectory()).filePath(targetSubdirectory);
    if (!QDir().mkpath(targetDirectory)) {
        QMessageBox::critical(this, "Import", "Failed to create " + targetSubdirectory + ".");
        return;
    }

    std::set<QString> copiedFiles;
    for (const QString& selectedFile : selectedFiles) {
        const QFileInfo sourceInfo(selectedFile);
        const QString baseName = sourceInfo.completeBaseName();
        const QDir sourceDir = sourceInfo.dir();
        QStringList candidates;
        const QString lowerFileName = sourceInfo.fileName().toLower();
        for (const QString& extension : companionExtensionsFor(sourceInfo.suffix())) {
            if (extension.contains('.')) {
                const QString sidecarName = sourceInfo.fileName() + "." + extension;
                if (sourceDir.exists(sidecarName)) {
                    candidates.push_back(sourceDir.filePath(sidecarName));
                }
                continue;
            }
            const QStringList matches = sourceDir.entryList({baseName + "." + extension}, QDir::Files);
            for (const QString& match : matches) {
                candidates.push_back(sourceDir.filePath(match));
            }
        }
        if (candidates.isEmpty()) {
            candidates.push_back(selectedFile);
        }

        for (const QString& candidate : candidates) {
            copiedFiles.insert(QFileInfo(candidate).absoluteFilePath());
        }

        if (lowerFileName.endsWith(".aux.xml")) {
            copiedFiles.insert(sourceInfo.absoluteFilePath());
        }
    }

    QString error;
    int copiedCount = 0;
    for (const QString& sourcePath : copiedFiles) {
        const QString targetPath = QDir(targetDirectory).filePath(QFileInfo(sourcePath).fileName());
        if (!copyFileReplacing(sourcePath, targetPath, &error)) {
            QMessageBox::critical(this, "Import", error);
            return;
        }
        ++copiedCount;
    }

    logGuiAction(QString("Imported %1 file(s) into %2").arg(copiedCount).arg(targetSubdirectory));
    refreshFileTree();
}

void MainWindow::stopActiveWork() {
    if (!runner_->isRunning()) {
        statusBar()->showMessage("No backend process is running.", 3000);
        return;
    }
    logGuiAction("Stop requested for the active backend process");
    runner_->stop();
}

void MainWindow::updateAuxiliaryPanelLayout() {
    if (!batchPanel_ || !boundaryCsvPanel_ || !rightSplitter_ || !rightTopTabs_ || !document_) {
        return;
    }
    const bool luwMode = document_->mode() == RunMode::Luw;
    rightTopTabs_->setTabVisible(0, !luwMode);
    rightTopTabs_->setTabVisible(1, luwMode);
    if (luwMode) {
        rightTopTabs_->setCurrentWidget(boundaryCsvPanel_);
    } else {
        rightTopTabs_->setCurrentWidget(batchPanel_);
    }
    rightSplitter_->setSizes({420, 280});
}

void MainWindow::logGuiAction(const QString& message) {
    if (!console_ || runner_->isRunning() || message.trimmed().isEmpty()) {
        return;
    }
    console_->appendText("[GUI] " + message.trimmed() + "\n");
}

void MainWindow::newDeck(RunMode mode, const QString& caseName) {
    document_->clearFilePath();
    document_->setMode(mode);
    document_->applyRawText(buildSkeletonText(mode, caseName));
    refreshEditors();
    refreshRawDeck();
}

QString MainWindow::buildSkeletonText(RunMode mode, const QString& caseName) const {
    QStringList lines;
    lines << "// LUW Studio generated starter deck";
    const QString resolvedCaseName = caseName.trimmed().isEmpty()
        ? QString("case_%1").arg(runModeDisplayName(mode).toLower())
        : caseName.trimmed();
    lines << QString("casename = %1").arg(resolvedCaseName);
    lines << QString("datetime = %1").arg(QDateTime::currentDateTimeUtc().toString("yyyyMMddhhmmss"));
    lines << "cut_lon_manual = [121.300000, 121.700000]";
    lines << "cut_lat_manual = [31.100000, 31.400000]";
    lines << "base_height = 50.000000";
    lines << "z_limit = 500.000000";
    lines << "si_x_cfd = [0.000000, 1000.000000]";
    lines << "si_y_cfd = [0.000000, 1000.000000]";
    lines << "si_z_cfd = [0.000000, 300.000000]";
    lines << "";
    lines << "// CFD control";
    lines << "n_gpu = [1, 1, 1]";
    lines << "mesh_control = \"gpu_memory\"";
    lines << "gpu_memory = 20000";
    lines << "cell_size = 10.000000";
    lines << "validation = pass";
    lines << "high_order = true";
    lines << "flux_correction = true";
    lines << "coriolis_term = true";
    lines << "buoyancy = false";
    lines << "";
    lines << "// Output and probes";
    lines << "unsteady_output = 0";
    lines << "probes_output = 0";
    lines << "purge_avg = 0";
    lines << "purge_avg_stride = 1";
    lines << "output_tke_ti_tls = [tke, ti, tls]";
    lines << "probes = ";
    lines << "";
    lines << "// Physics";
    lines << "enable_buffer_nudging = false";
    lines << "buffer_thickness_m = 0.0";
    lines << "buffer_tau_s = 0.0";
    lines << "buffer_nudge_vertical = false";
    lines << "enable_top_sponge = false";
    lines << "sponge_thickness_m = 0.0";
    lines << "sponge_tau_s = 0.0";
    lines << "sponge_ref_mode = mode0";
    lines << "";
    lines << "// VK inlet";
    lines << "vk_inlet_enable = false";
    lines << "vk_inlet_ti = 0.10";
    lines << "vk_inlet_sigma = 0.0";
    lines << "vk_inlet_l = 0.0";
    lines << "vk_inlet_nmodes = 128";
    lines << "vk_inlet_seed = 1469598103934665603";
    lines << "vk_inlet_update_stride = 1";
    lines << "vk_inlet_uc_mode = NORMAL_COMPONENT";
    lines << "vk_inlet_same_realization_all_faces = false";
    lines << "vk_inlet_stride_interpolation = false";
    lines << "vk_inlet_inflow_only = true";
    lines << "vk_inlet_anisotropy = [1.000000, 1.000000, 1.000000]";
    if (mode == RunMode::Luwdg || mode == RunMode::Luwpf) {
        lines << "";
        lines << "// Batch";
        lines << "x_exp_rat = 5.000000";
        lines << "y_exp_rat = 5.000000";
        lines << "angle = [0.000000, 90.000000, 180.000000, 270.000000]";
    }
    if (mode == RunMode::Luwdg) {
        lines << "inflow = [5.000000, 10.000000]";
    }
    return lines.join('\n') + '\n';
}

void MainWindow::refreshEditors() {
    if (workflowSummary_) {
        workflowSummary_->setText(
            QString("Mode: %1\nDeck: %2\nProject: %3\nRESULTS: %4")
                .arg(runModeDisplayName(document_->mode()))
                .arg(document_->filePath().isEmpty() ? "<unsaved>" : document_->filePath())
                .arg(document_->projectDirectory())
                .arg(document_->resultsDirectory()));
    }

    for (const EditorBinding& binding : bindings_) {
        writeEditorValue(binding, document_->typedValue(binding.spec.key));
    }
}

void MainWindow::refreshRawDeck() {
    if (!rawDeckEdit_) {
        return;
    }
    const QString rendered = document_->renderedText();
    if (rawDeckEdit_->toPlainText() == rendered) {
        return;
    }
    rawDeckEdit_->blockSignals(true);
    rawDeckEdit_->setPlainText(rendered);
    rawDeckEdit_->blockSignals(false);
}

void MainWindow::refreshFileTree() {
    if (!fileModel_ || !fileTree_) {
        return;
    }
    fileModel_->setRootPath(document_->projectDirectory());
    fileTree_->setRootIndex(fileModel_->index(document_->projectDirectory()));
}

void MainWindow::setCurrentPage(const QString& pageId) {
    const auto it = pageById_.constFind(pageId);
    if (it == pageById_.cend()) {
        return;
    }
    sectionStack_->setCurrentIndex(it.value());
}

void MainWindow::runPreset(CommandPreset preset) {
    QString error;
    if (document_->filePath().isEmpty()) {
        const QString path = QFileDialog::getSaveFileName(
            this,
            "Save project before running",
            QDir(document_->projectDirectory()).filePath(defaultDeckName(document_->mode())),
            "LUW Deck (*.luw *.luwdg *.luwpf)");
        if (path.isEmpty()) {
            return;
        }
        if (!document_->saveAs(path, &error)) {
            QMessageBox::critical(this, "Save project", error);
            return;
        }
    } else if (!document_->save(&error)) {
        QMessageBox::critical(this, "Save project", error);
        return;
    }
    runner_->startPreset(preset);
}

void MainWindow::runCutVis() {
    QString error;
    if (document_->filePath().isEmpty()) {
        const QString path = QFileDialog::getSaveFileName(
            this,
            "Save project before running cut_vis",
            QDir(document_->projectDirectory()).filePath(defaultDeckName(document_->mode())),
            "LUW Deck (*.luw *.luwdg *.luwpf)");
        if (path.isEmpty()) {
            return;
        }
        if (!document_->saveAs(path, &error)) {
            QMessageBox::critical(this, "Save project", error);
            return;
        }
    } else if (!document_->save(&error)) {
        QMessageBox::critical(this, "Save project", error);
        return;
    }
    runner_->startPreset(CommandPreset::CutVis, buildCutVisArguments());
}

QStringList MainWindow::buildCutVisArguments() const {
    QStringList args;
    const QString vtkPath = latestResultVtk();
    if (!vtkPath.isEmpty()) {
        args << vtkPath;
    }
    args << "--data-dir" << document_->resultsDirectory();
    args << "--output-dir" << QDir(document_->resultsDirectory()).filePath("cut_vis");

    if (document_->hasKey("cell_size")) {
        args << "--cell-size" << QString::number(document_->typedValue("cell_size").toDouble());
    }
    if (document_->hasKey("cut_lon_manual")) {
        const QVariantList lon = document_->typedValue("cut_lon_manual").toList();
        if (lon.size() >= 2) {
            args << "--domain-lon-min" << QString::number(lon[0].toDouble());
            args << "--domain-lon-max" << QString::number(lon[1].toDouble());
        }
    }
    if (document_->hasKey("cut_lat_manual")) {
        const QVariantList lat = document_->typedValue("cut_lat_manual").toList();
        if (lat.size() >= 2) {
            args << "--domain-lat-min" << QString::number(lat[0].toDouble());
            args << "--domain-lat-max" << QString::number(lat[1].toDouble());
        }
    }
    if (document_->hasKey("utm_crs")) {
        args << "--utm-crs" << document_->typedValue("utm_crs").toString();
    }
    if (document_->hasKey("rotate_deg")) {
        args << "--rotate-deg" << QString::number(document_->typedValue("rotate_deg").toDouble());
    }

    if (cutVisExtraArgsEdit_ && !cutVisExtraArgsEdit_->text().trimmed().isEmpty()) {
        args << cutVisExtraArgsEdit_->text().split(QRegularExpression(R"(\s+)"), Qt::SkipEmptyParts);
    }
    return args;
}

QString MainWindow::latestResultVtk() const {
    QDir solverVtkDir(QDir(document_->resultsDirectory()).filePath("vtk"));
    if (solverVtkDir.exists()) {
        QFileInfoList avgFiles = solverVtkDir.entryInfoList({"*_avg-*.vtk"}, QDir::Files, QDir::Time);
        if (!avgFiles.isEmpty()) {
            return avgFiles.front().absoluteFilePath();
        }
        QFileInfoList allSolverFiles = solverVtkDir.entryInfoList({"*.vtk"}, QDir::Files, QDir::Time);
        if (!allSolverFiles.isEmpty()) {
            return allSolverFiles.front().absoluteFilePath();
        }
    }

    QDir resultsDir(document_->resultsDirectory());
    QFileInfoList files = resultsDir.entryInfoList({"*.vtk"}, QDir::Files, QDir::Time);
    if (files.isEmpty()) {
        return {};
    }
    return files.front().absoluteFilePath();
}

void MainWindow::loadLatestResult() {
    QString error;
    if (!vtkView_ || !vtkView_->loadLatestResult(&error)) {
        statusBar()->showMessage(error.isEmpty() ? "No VTK file found in RESULTS/vtk." : error, 5000);
        return;
    }
}

void MainWindow::loadViewerFile(const QString& filePath) {
    QString error;
    if (!vtkView_->loadFile(filePath, &error)) {
        QMessageBox::critical(this, "Load visualization file", error);
        return;
    }
    if (filePath.endsWith(".vtk", Qt::CaseInsensitive) && wavenumberPanel_) {
        wavenumberPanel_->setSuggestedFilePath(filePath);
    }
}

QVariant MainWindow::readEditorValue(const EditorBinding& binding) const {
    switch (binding.spec.kind) {
    case FieldKind::Boolean:
        return qobject_cast<QCheckBox*>(binding.editor)->isChecked();
    case FieldKind::Enum:
        return qobject_cast<QComboBox*>(binding.editor)->currentText();
    case FieldKind::Integer:
        return qobject_cast<QSpinBox*>(binding.editor)->value();
    case FieldKind::Float:
        return qobject_cast<QDoubleSpinBox*>(binding.editor)->value();
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::FloatList:
        return parseLooseList(qobject_cast<QLineEdit*>(binding.editor)->text(), false);
    case FieldKind::UIntTriplet:
        return parseLooseList(qobject_cast<QLineEdit*>(binding.editor)->text(), true);
    case FieldKind::TokenList: {
        const QStringList tokens = qobject_cast<QLineEdit*>(binding.editor)->text().split(QRegularExpression(R"([,\s]+)"), Qt::SkipEmptyParts);
        return tokens;
    }
    case FieldKind::Multiline:
        return qobject_cast<QPlainTextEdit*>(binding.editor)->toPlainText();
    case FieldKind::String:
        return qobject_cast<QLineEdit*>(binding.editor)->text();
    }
    return {};
}

void MainWindow::writeEditorValue(const EditorBinding& binding, const QVariant& value) {
    switch (binding.spec.kind) {
    case FieldKind::Boolean: {
        auto* widget = qobject_cast<QCheckBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setChecked(value.toBool());
        break;
    }
    case FieldKind::Enum: {
        auto* widget = qobject_cast<QComboBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        const int index = widget->findText(value.toString());
        if (index >= 0) {
            widget->setCurrentIndex(index);
        }
        break;
    }
    case FieldKind::Integer: {
        auto* widget = qobject_cast<QSpinBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setValue(value.toInt());
        break;
    }
    case FieldKind::Float: {
        auto* widget = qobject_cast<QDoubleSpinBox*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setValue(value.toDouble());
        break;
    }
    case FieldKind::FloatPair:
    case FieldKind::FloatTriplet:
    case FieldKind::FloatList: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(renderVariantList(value.toList(), false));
        break;
    }
    case FieldKind::UIntTriplet: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(renderVariantList(value.toList(), true));
        break;
    }
    case FieldKind::TokenList: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(value.toStringList().join(", "));
        break;
    }
    case FieldKind::Multiline: {
        auto* widget = qobject_cast<QPlainTextEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setPlainText(value.toString());
        break;
    }
    case FieldKind::String: {
        auto* widget = qobject_cast<QLineEdit*>(binding.editor);
        const QSignalBlocker blocker(widget);
        widget->setText(value.toString());
        break;
    }
    }
}

}
