/****************************************************************************
** Meta object code from reading C++ file 'PlotWidgets.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../include/luwgui/PlotWidgets.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PlotWidgets.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS = QtMocHelpers::stringData(
    "luwgui::ExportablePlotWidget"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

Q_CONSTINIT const QMetaObject luwgui::ExportablePlotWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<ExportablePlotWidget, std::true_type>
    >,
    nullptr
} };

void luwgui::ExportablePlotWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

const QMetaObject *luwgui::ExportablePlotWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *luwgui::ExportablePlotWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSluwguiSCOPEExportablePlotWidgetENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int luwgui::ExportablePlotWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    return _id;
}
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS = QtMocHelpers::stringData(
    "luwgui::ProfilePlotWidget"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

Q_CONSTINIT const QMetaObject luwgui::ProfilePlotWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<ExportablePlotWidget::staticMetaObject>(),
    qt_meta_stringdata_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<ProfilePlotWidget, std::true_type>
    >,
    nullptr
} };

void luwgui::ProfilePlotWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

const QMetaObject *luwgui::ProfilePlotWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *luwgui::ProfilePlotWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSluwguiSCOPEProfilePlotWidgetENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return ExportablePlotWidget::qt_metacast(_clname);
}

int luwgui::ProfilePlotWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExportablePlotWidget::qt_metacall(_c, _id, _a);
    return _id;
}
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS = QtMocHelpers::stringData(
    "luwgui::SpectrumPlotWidget"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

Q_CONSTINIT const QMetaObject luwgui::SpectrumPlotWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<ExportablePlotWidget::staticMetaObject>(),
    qt_meta_stringdata_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<SpectrumPlotWidget, std::true_type>
    >,
    nullptr
} };

void luwgui::SpectrumPlotWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

const QMetaObject *luwgui::SpectrumPlotWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *luwgui::SpectrumPlotWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSluwguiSCOPESpectrumPlotWidgetENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return ExportablePlotWidget::qt_metacast(_clname);
}

int luwgui::SpectrumPlotWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExportablePlotWidget::qt_metacall(_c, _id, _a);
    return _id;
}
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS = QtMocHelpers::stringData(
    "luwgui::HeatmapPlotWidget"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

Q_CONSTINIT const QMetaObject luwgui::HeatmapPlotWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<ExportablePlotWidget::staticMetaObject>(),
    qt_meta_stringdata_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<HeatmapPlotWidget, std::true_type>
    >,
    nullptr
} };

void luwgui::HeatmapPlotWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

const QMetaObject *luwgui::HeatmapPlotWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *luwgui::HeatmapPlotWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSluwguiSCOPEHeatmapPlotWidgetENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return ExportablePlotWidget::qt_metacast(_clname);
}

int luwgui::HeatmapPlotWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExportablePlotWidget::qt_metacall(_c, _id, _a);
    return _id;
}
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS_t {};
constexpr auto qt_meta_stringdata_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS = QtMocHelpers::stringData(
    "luwgui::DistributionPlotWidget"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

Q_CONSTINIT const QMetaObject luwgui::DistributionPlotWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<ExportablePlotWidget::staticMetaObject>(),
    qt_meta_stringdata_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<DistributionPlotWidget, std::true_type>
    >,
    nullptr
} };

void luwgui::DistributionPlotWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    (void)_o;
    (void)_id;
    (void)_c;
    (void)_a;
}

const QMetaObject *luwgui::DistributionPlotWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *luwgui::DistributionPlotWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSluwguiSCOPEDistributionPlotWidgetENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return ExportablePlotWidget::qt_metacast(_clname);
}

int luwgui::DistributionPlotWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = ExportablePlotWidget::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
