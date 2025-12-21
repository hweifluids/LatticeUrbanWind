import os
import sys
import time
import errno
import operator
from dataclasses import dataclass
from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    section: str
    ftype: str
    quoted: bool = False
    list_len: int = 0
    elem_type: str = ""


SCHEMA_ORDER = [
    FieldSpec("casename", "Case name", "Case Setups", "str", quoted=False),
    FieldSpec("cut_lon_manual", "Longitude range", "Case Setups", "list", list_len=2, elem_type="float"),
    FieldSpec("cut_lat_manual", "Latitude range", "Case Setups", "list", list_len=2, elem_type="float"),
    FieldSpec("datetime", "Date Time code", "Case Setups", "int_strict_digits", quoted=False),
    FieldSpec("low_altitude", "Low altitude range", "Case Setups", "int", quoted=False),

    FieldSpec("n_gpu", "Multi GPU ultilization", "Numerical Control", "list", list_len=3, elem_type="int"),
    FieldSpec("mesh_control", "Mesh size control", "Numerical Control", "mesh_control", quoted=True),
    FieldSpec("gpu_memory", "vRAM (MB)", "Numerical Control", "int", quoted=False),
    FieldSpec("cell_size", "Base size (m)", "Numerical Control", "int", quoted=False),
    FieldSpec("validation", "Coordinate validation", "Numerical Control", "str", quoted=False),
    FieldSpec("high_order", "High order interpolation", "Numerical Control", "bool", quoted=False),
    FieldSpec("flux_correction", "Flux correction", "Numerical Control", "bool", quoted=False),
    FieldSpec("coriolis_term", "Coriolis source term", "Numerical Control", "bool", quoted=False),
    FieldSpec("research_output", "N Snapshots for research", "Numerical Control", "int", quoted=False),

    FieldSpec("si_x_cfd", "CFD domain: X", "Internal Information", "list", list_len=2, elem_type="float"),
    FieldSpec("si_y_cfd", "CFD domain: Y", "Internal Information", "list", list_len=2, elem_type="float"),
    FieldSpec("si_z_cfd", "CFD domain: Z", "Internal Information", "list", list_len=2, elem_type="float"),

    FieldSpec("utm_crs", "UTM CRS", "Internal Information", "str", quoted=True),
    FieldSpec("rotate_deg", "Rotational correction degree", "Internal Information", "float", quoted=False),
    FieldSpec("origin_shift_applied", "Origional shifting", "Internal Information", "bool", quoted=False),

    FieldSpec("um_vol", "Volume mean wind", "Internal Information", "list", list_len=3, elem_type="float"),
    FieldSpec("um_bc", "Boundary mean wind", "Internal Information", "list", list_len=3, elem_type="float"),
    FieldSpec("downstream_bc", "Downstream boundary", "Internal Information", "str", quoted=True),
    FieldSpec("downstream_bc_yaw", "Downstream boundary angle", "Internal Information", "float", quoted=False),
]

SCHEMA_BY_KEY = {s.key: s for s in SCHEMA_ORDER}
KNOWN_KEYS = set(SCHEMA_BY_KEY.keys())


def strip_inline_comment(line):
    idx = line.find("//")
    if idx < 0:
        return line, ""
    return line[:idx], line[idx:]


def parse_bool(s):
    t = s.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    return None


def parse_number(s, want):
    t = s.strip()
    if not t:
        return None
    try:
        if want == "int":
            return int(t)
        if want == "float":
            return float(t)
    except ValueError:
        return None
    return None


def parse_string(s):
    t = s.strip()
    if len(t) >= 2 and t[0] == '"' and t[operator.sub(len(t), 1)] == '"':
        return t[1:operator.sub(len(t), 1)]
    return t


def parse_list(s, elem_type, expected_len):
    t = s.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    inner = t[1:operator.sub(len(t), 1)].strip()
    if inner == "":
        parts = []
    else:
        parts = [p.strip() for p in inner.split(",")]
    if expected_len > 0 and len(parts) != expected_len:
        return None

    out = []
    for p in parts:
        if elem_type == "int":
            v = parse_number(p, "int")
            if v is None:
                return None
            out.append(v)
        elif elem_type == "float":
            v = parse_number(p, "float")
            if v is None:
                return None
            out.append(v)
        else:
            return None
    return out


def format_value(key, value):
    spec = SCHEMA_BY_KEY[key]
    if value is None:
        return ""

    if spec.ftype == "bool":
        return "true" if bool(value) else "false"

    if spec.ftype in ("int", "int_strict_digits"):
        return str(int(value))

    if spec.ftype == "float":
        return f"{float(value):.6f}"

    if spec.ftype == "mesh_control":
        s = str(value)
        if spec.quoted:
            return f"\"{s}\""
        return s

    if spec.ftype == "str":
        s = str(value)
        if spec.quoted:
            return f"\"{s}\""
        return s

    if spec.ftype == "list":
        if not isinstance(value, list):
            return ""
        parts = []
        for v in value:
            if spec.elem_type == "int":
                parts.append(str(int(v)))
            elif spec.elem_type == "float":
                parts.append(f"{float(v):.6f}")
            else:
                parts.append(str(v))
        return "[ " + ", ".join(parts) + " ]"

    return str(value)


def normalize_cache_value(key, parsed):
    spec = SCHEMA_BY_KEY[key]
    if parsed is None:
        return None

    if spec.ftype == "bool":
        return bool(parsed)

    if spec.ftype in ("int", "int_strict_digits"):
        try:
            return int(parsed)
        except Exception:
            return None

    if spec.ftype == "float":
        try:
            return float(parsed)
        except Exception:
            return None

    if spec.ftype == "mesh_control":
        s = str(parsed)
        if s not in ("gpu_memory", "cell_size"):
            return None
        return s

    if spec.ftype == "str":
        return str(parsed)

    if spec.ftype == "list":
        if not isinstance(parsed, list):
            return None
        if spec.list_len and len(parsed) != spec.list_len:
            return None
        if spec.elem_type == "int":
            try:
                return [int(x) for x in parsed]
            except Exception:
                return None
        if spec.elem_type == "float":
            try:
                return [float(x) for x in parsed]
            except Exception:
                return None
        return None

    return parsed


def parse_file_text(text):
    lines = text.splitlines(True)
    values = {}
    found_keys = set()

    for raw in lines:
        content, _comment = strip_inline_comment(raw)
        line = content.strip()
        if not line:
            continue
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip()
        if key not in KNOWN_KEYS:
            continue

        spec = SCHEMA_BY_KEY[key]
        val_raw = right.strip()

        parsed = None
        if spec.ftype in ("str", "mesh_control"):
            parsed = parse_string(val_raw)
        elif spec.ftype == "bool":
            parsed = parse_bool(val_raw)
        elif spec.ftype in ("int", "int_strict_digits"):
            parsed = parse_number(val_raw, "int")
        elif spec.ftype == "float":
            parsed = parse_number(val_raw, "float")
        elif spec.ftype == "list":
            parsed = parse_list(val_raw, spec.elem_type, spec.list_len)

        values[key] = normalize_cache_value(key, parsed)
        found_keys.add(key)

    return values, lines, found_keys


class IntDigitsValidator(QtGui.QValidator):
    def __init__(self, allow_empty, parent=None):
        super().__init__(parent)
        self.allow_empty = allow_empty

    def validate(self, input_str, pos):
        if input_str == "":
            if self.allow_empty:
                return QtGui.QValidator.Intermediate, input_str, pos
            return QtGui.QValidator.Invalid, input_str, pos
        if input_str.isdigit():
            return QtGui.QValidator.Acceptable, input_str, pos
        if all(ch.isdigit() for ch in input_str):
            return QtGui.QValidator.Intermediate, input_str, pos
        return QtGui.QValidator.Invalid, input_str, pos


class SignedIntValidator(QtGui.QValidator):
    def __init__(self, allow_empty, parent=None):
        super().__init__(parent)
        self.allow_empty = allow_empty
        self.minus = chr(45)
        self.plus = chr(43)

    def validate(self, input_str, pos):
        if input_str == "":
            if self.allow_empty:
                return QtGui.QValidator.Intermediate, input_str, pos
            return QtGui.QValidator.Invalid, input_str, pos

        s = input_str
        if s[0] in (self.minus, self.plus):
            if len(s) == 1:
                return QtGui.QValidator.Intermediate, input_str, pos
            s = s[1:len(s)]

        if s.isdigit():
            return QtGui.QValidator.Acceptable, input_str, pos
        if all(ch.isdigit() for ch in s):
            return QtGui.QValidator.Intermediate, input_str, pos
        return QtGui.QValidator.Invalid, input_str, pos


class SignedFloatValidator(QtGui.QValidator):
    def __init__(self, allow_empty, parent=None):
        super().__init__(parent)
        self.allow_empty = allow_empty
        self.minus = chr(45)
        self.plus = chr(43)
        self.dot = "."

    def validate(self, input_str, pos):
        if input_str == "":
            if self.allow_empty:
                return QtGui.QValidator.Intermediate, input_str, pos
            return QtGui.QValidator.Invalid, input_str, pos

        s = input_str.strip()
        if s == "":
            if self.allow_empty:
                return QtGui.QValidator.Intermediate, input_str, pos
            return QtGui.QValidator.Invalid, input_str, pos

        if s[0] in (self.minus, self.plus):
            if len(s) == 1:
                return QtGui.QValidator.Intermediate, input_str, pos
            s = s[1:len(s)]

        if s.count(self.dot) > 1:
            return QtGui.QValidator.Invalid, input_str, pos

        allowed = set("0123456789.")
        if any(ch not in allowed for ch in s):
            return QtGui.QValidator.Invalid, input_str, pos

        if s == ".":
            return QtGui.QValidator.Intermediate, input_str, pos

        if s.endswith("."):
            left = s[0:operator.sub(len(s), 1)]
            if left.isdigit():
                return QtGui.QValidator.Intermediate, input_str, pos
            return QtGui.QValidator.Invalid, input_str, pos

        if "." in s:
            left, right = s.split(".", 1)
            if left == "" and right == "":
                return QtGui.QValidator.Intermediate, input_str, pos
            if left != "" and not left.isdigit():
                return QtGui.QValidator.Invalid, input_str, pos
            if right != "" and not right.isdigit():
                return QtGui.QValidator.Invalid, input_str, pos
            if (left == "" and right.isdigit()) or (left.isdigit() and right == ""):
                return QtGui.QValidator.Intermediate, input_str, pos
            if left.isdigit() and right.isdigit():
                return QtGui.QValidator.Acceptable, input_str, pos
            return QtGui.QValidator.Intermediate, input_str, pos

        if s.isdigit():
            return QtGui.QValidator.Acceptable, input_str, pos

        return QtGui.QValidator.Intermediate, input_str, pos


class IndicatorLight(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QtGui.QColor(120, 120, 120)
        self.token = 0
        self.setFixedSize(14, 14)

    def set_color(self, color):
        self.color = color
        self.update()

    def flash(self, color, duration_ms):
        self.token += 1
        token = self.token
        self.set_color(color)

        def clear_if_current():
            if token == self.token:
                self.set_color(QtGui.QColor(120, 120, 120))

        QtCore.QTimer.singleShot(duration_ms, clear_if_current)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w = operator.sub(self.width(), 2)
        h = operator.sub(self.height(), 2)
        rect = QtCore.QRectF(1.0, 1.0, float(w), float(h))
        pen = QtGui.QPen(QtGui.QColor(60, 60, 60))
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(self.color))
        painter.drawEllipse(rect)


class IntLineEdit(QtWidgets.QLineEdit):
    valueEdited = QtCore.pyqtSignal()

    def __init__(self, strict_digits, parent=None):
        super().__init__(parent)
        if strict_digits:
            self.setValidator(IntDigitsValidator(True, self))
        else:
            self.setValidator(SignedIntValidator(True, self))
        self.textEdited.connect(self.valueEdited.emit)

    def get_value(self):
        t = self.text().strip()
        if t == "":
            return None
        try:
            return int(t)
        except ValueError:
            return None

    def set_value(self, v):
        if v is None:
            self.setText("")
        else:
            self.setText(str(int(v)))


class FloatLineEdit(QtWidgets.QLineEdit):
    valueEdited = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setValidator(SignedFloatValidator(True, self))
        self.textEdited.connect(self.valueEdited.emit)

    def get_value(self):
        t = self.text().strip()
        if t == "":
            return None
        try:
            return float(t)
        except ValueError:
            return None

    def set_value(self, v):
        if v is None:
            self.setText("")
        else:
            self.setText(f"{float(v):.6f}")


class BoolCombo(QtWidgets.QComboBox):
    valueEdited = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("")
        self.addItem("true")
        self.addItem("false")
        self.currentIndexChanged.connect(self.valueEdited.emit)

    def get_value(self):
        t = self.currentText().strip().lower()
        if t == "true":
            return True
        if t == "false":
            return False
        return None

    def set_value(self, v):
        if v is None:
            self.setCurrentIndex(0)
        else:
            self.setCurrentText("true" if bool(v) else "false")


class MeshControlCombo(QtWidgets.QComboBox):
    valueEdited = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("")
        self.addItem("vRAM (MB)", userData="gpu_memory")
        self.addItem("Base size (m)", userData="cell_size")
        self.currentIndexChanged.connect(self.valueEdited.emit)

    def get_value(self):
        data = self.currentData()
        if data in ("gpu_memory", "cell_size"):
            return str(data)
        return None

    def set_value(self, v):
        if v is None:
            self.setCurrentIndex(0)
            return
        if v == "gpu_memory":
            self.setCurrentIndex(1)
        elif v == "cell_size":
            self.setCurrentIndex(2)
        else:
            self.setCurrentIndex(0)


class ListEdit(QtWidgets.QWidget):
    valueEdited = QtCore.pyqtSignal()

    def __init__(self, elem_type, length, parent=None):
        super().__init__(parent)
        self.elem_type = elem_type
        self.length = length
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.edits = []
        for _i in range(length):
            if elem_type == "int":
                e = IntLineEdit(False, parent=self)
                e.valueEdited.connect(self.valueEdited.emit)
                self.edits.append(e)
                layout.addWidget(e)
            else:
                e = FloatLineEdit(parent=self)
                e.valueEdited.connect(self.valueEdited.emit)
                self.edits.append(e)
                layout.addWidget(e)

    def get_value(self):
        vals = []
        for e in self.edits:
            if isinstance(e, IntLineEdit):
                v = e.get_value()
                if v is None:
                    return None
                vals.append(v)
            elif isinstance(e, FloatLineEdit):
                v = e.get_value()
                if v is None:
                    return None
                vals.append(v)
            else:
                return None
        return vals

    def set_value(self, v):
        if v is None or (not isinstance(v, list)) or len(v) != self.length:
            for e in self.edits:
                e.setText("")
            return
        for e, x in zip(self.edits, v):
            if isinstance(e, IntLineEdit):
                e.set_value(x)
            elif isinstance(e, FloatLineEdit):
                e.set_value(x)


class ConflictDialog(QtWidgets.QDialog):
    def __init__(self, parent, conflicts):
        super().__init__(parent)
        self.setWindowTitle("Conflict detected")
        self.setModal(True)
        self.resize(980, 420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)

        info = QtWidgets.QLabel(
            "The file changed on disk and some fields were edited in the cache.\n"
            "Choose which value to keep for each conflicting field."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Section", "Field", "Cache", "File", "Keep"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setRowCount(len(conflicts))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        for row, c in enumerate(conflicts):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(c["section"]))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(c["label"]))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(c["cache"]))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(c["file"]))

            combo = QtWidgets.QComboBox(self.table)
            combo.addItem("Keep cache")
            combo.addItem("Use file")
            combo.setCurrentIndex(0)
            self.table.setCellWidget(row, 4, combo)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        buttons = QtWidgets.QDialogButtonBox(self)
        buttons.setStandardButtons(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choices(self):
        out = {}
        for row in range(self.table.rowCount()):
            section = self.table.item(row, 0).text()
            label = self.table.item(row, 1).text()
            key = f"{section}::{label}"
            combo = self.table.cellWidget(row, 4)
            out[key] = combo.currentText()
        return out


class SectionBox(QtWidgets.QWidget):
    def __init__(self, title, right_widget=None, parent=None):
        super().__init__(parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        lbl = QtWidgets.QLabel(title)
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
        header.addWidget(lbl, 1)

        if right_widget is not None:
            header.addWidget(right_widget, 0)

        outer.addLayout(header)

        self.frame = QtWidgets.QFrame()
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)

        inner = QtWidgets.QVBoxLayout(self.frame)
        inner.setContentsMargins(8, 6, 8, 6)
        inner.setSpacing(2)

        self.form = QtWidgets.QFormLayout()
        self.form.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.form.setFormAlignment(QtCore.Qt.AlignTop)
        self.form.setVerticalSpacing(2)
        self.form.setHorizontalSpacing(6)
        inner.addLayout(self.form)

        outer.addWidget(self.frame)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUW Config Editor")
        self.resize(1020, 820)

        self.file_path = ""
        self.last_stat = None
        self.last_lines = []
        self.found_keys = set()

        self.baseline = {}
        self.cache = {}
        self.dirty = set()

        self.widgets_by_key = {}
        self.internal_editable = False
        self.syncing_ui = False

        self.build_ui()
        self.build_timer()

    def build_ui(self):
        QtWidgets.QApplication.setStyle("Fusion")

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        self.path_label = QtWidgets.QLabel("No file selected")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.path_label, 1)

        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self.open_dialog)
        top.addWidget(self.open_btn)
        root.addLayout(top)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)

        container = QtWidgets.QWidget()
        self.scroll.setWidget(container)
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        self.case_box = SectionBox("Case Setups")
        vbox.addWidget(self.case_box)

        self.num_box = SectionBox("Numerical Control")
        vbox.addWidget(self.num_box)

        self.internal_lock_btn = QtWidgets.QToolButton()
        self.internal_lock_btn.setCheckable(True)
        self.internal_lock_btn.setChecked(False)
        self.internal_lock_btn.setText("Unlock")
        self.internal_lock_btn.clicked.connect(self.toggle_internal_lock)

        self.internal_box = SectionBox("Internal Information", right_widget=self.internal_lock_btn)
        vbox.addWidget(self.internal_box)

        vbox.addStretch(1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(6)

        self.indicator = IndicatorLight()
        bottom.addWidget(self.indicator)

        bottom.addStretch(1)

        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.reload_btn.clicked.connect(self.reload_from_disk)
        bottom.addWidget(self.reload_btn)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_to_disk)
        bottom.addWidget(self.save_btn)

        root.addLayout(bottom)

        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)
        self.status.showMessage("Ready", 2000)

        self.build_fields()
        self.apply_internal_enabled(False)
        self.apply_mesh_control_enabled()

    def build_fields(self):
        for spec in SCHEMA_ORDER:
            w = self.create_widget_for_spec(spec)
            self.widgets_by_key[spec.key] = w
            label = QtWidgets.QLabel(spec.label)

            if spec.section == "Case Setups":
                self.case_box.form.addRow(label, w)
            elif spec.section == "Numerical Control":
                self.num_box.form.addRow(label, w)
            else:
                self.internal_box.form.addRow(label, w)

    def create_widget_for_spec(self, spec):
        if spec.ftype == "bool":
            w = BoolCombo()
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            return w

        if spec.ftype == "mesh_control":
            w = MeshControlCombo()
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            w.valueEdited.connect(self.apply_mesh_control_enabled)
            return w

        if spec.ftype == "int":
            w = IntLineEdit(False)
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            return w

        if spec.ftype == "int_strict_digits":
            w = IntLineEdit(True)
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            return w

        if spec.ftype == "float":
            w = FloatLineEdit()
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            return w

        if spec.ftype == "str":
            w = QtWidgets.QLineEdit()
            w.textEdited.connect(lambda _t, k=spec.key: self.on_user_edit(k))
            return w

        if spec.ftype == "list":
            w = ListEdit(spec.elem_type, spec.list_len)
            w.valueEdited.connect(lambda k=spec.key: self.on_user_edit(k))
            return w

        w = QtWidgets.QLineEdit()
        w.textEdited.connect(lambda _t, k=spec.key: self.on_user_edit(k))
        return w

    def build_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll_file)
        self.timer.start()

    def flash_yellow(self):
        self.indicator.flash(QtGui.QColor(230, 180, 20), 200)

    def flash_green(self):
        self.indicator.flash(QtGui.QColor(50, 180, 90), 2000)

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open configuration file",
            "",
            "LUW files (*.luw);;All files (*)",
        )
        if not path:
            return
        self.open_path(path)

    def open_path(self, path):
        self.file_path = path
        self.path_label.setText(path)
        ok, err = self.load_from_disk(True)
        if ok:
            self.status.showMessage("Loaded", 2000)
            self.flash_green()
        else:
            self.status.showMessage(err, 3000)
        self.update_title()

    def reload_from_disk(self):
        if not self.file_path:
            QtWidgets.QMessageBox.information(self, "No file", "Select a file first.")
            return
        ok, err = self.load_from_disk(True)
        if ok:
            self.status.showMessage("Reloaded", 2000)
            self.flash_green()
        else:
            self.status.showMessage(err, 3000)
        self.update_title()

    def update_title(self):
        name = os.path.basename(self.file_path) if self.file_path else "LUW Config Editor"
        mark = "*" if self.dirty else ""
        self.setWindowTitle(f"{name}{mark}")

    def stat_file(self):
        try:
            st = os.stat(self.file_path)
            return st.st_mtime, st.st_size
        except OSError:
            return None

    def read_text(self):
        try:
            with open(self.file_path, "r", encoding="utf8", errors="replace") as f:
                return True, f.read()
        except OSError as e:
            return False, str(e)

    def load_from_disk(self, discard_dirty):
        ok, txt_or_err = self.read_text()
        if not ok:
            return False, f"Read failed: {txt_or_err}"

        values, lines, found = parse_file_text(txt_or_err)
        self.last_lines = lines
        self.found_keys = found

        base = {}
        for spec in SCHEMA_ORDER:
            base[spec.key] = values.get(spec.key, None)

        self.baseline = dict(base)
        if discard_dirty:
            self.cache = dict(base)
            self.dirty.clear()
        else:
            for k in base.keys():
                if k not in self.dirty:
                    self.cache[k] = base[k]

        self.last_stat = self.stat_file()
        self.sync_ui_from_cache()
        return True, ""

    def sync_ui_from_cache(self):
        self.syncing_ui = True
        try:
            for spec in SCHEMA_ORDER:
                key = spec.key
                w = self.widgets_by_key[key]
                v = self.cache.get(key, None)
                self.set_widget_value(w, v)
                self.set_dirty_style(key, key in self.dirty)
            self.apply_mesh_control_enabled()
            self.update_title()
        finally:
            self.syncing_ui = False

    def set_widget_value(self, w, v):
        if isinstance(w, BoolCombo):
            w.set_value(v)
            return
        if isinstance(w, MeshControlCombo):
            w.set_value(v)
            return
        if isinstance(w, IntLineEdit):
            w.set_value(v)
            return
        if isinstance(w, FloatLineEdit):
            w.set_value(v)
            return
        if isinstance(w, ListEdit):
            w.set_value(v)
            return
        if isinstance(w, QtWidgets.QLineEdit):
            w.setText("" if v is None else str(v))
            return

    def get_widget_value(self, w):
        if isinstance(w, BoolCombo):
            return w.get_value()
        if isinstance(w, MeshControlCombo):
            return w.get_value()
        if isinstance(w, IntLineEdit):
            return w.get_value()
        if isinstance(w, FloatLineEdit):
            return w.get_value()
        if isinstance(w, ListEdit):
            return w.get_value()
        if isinstance(w, QtWidgets.QLineEdit):
            return w.text()
        return None

    def set_dirty_style(self, key, is_dirty):
        w = self.widgets_by_key[key]
        if is_dirty:
            w.setStyleSheet("border: 1px solid rgb(200, 155, 0);")
        else:
            w.setStyleSheet("")

    def is_internal_key(self, key):
        return SCHEMA_BY_KEY[key].section == "Internal Information"

    def on_user_edit(self, key):
        if self.syncing_ui:
            return

        if self.is_internal_key(key) and not self.internal_editable:
            self.syncing_ui = True
            try:
                w = self.widgets_by_key[key]
                self.set_widget_value(w, self.cache.get(key, None))
            finally:
                self.syncing_ui = False
            return

        w = self.widgets_by_key[key]
        new_v = normalize_cache_value(key, self.get_widget_value(w))
        self.cache[key] = new_v

        if self.baseline.get(key, None) != new_v:
            self.dirty.add(key)
            self.set_dirty_style(key, True)
        else:
            if key in self.dirty:
                self.dirty.remove(key)
            self.set_dirty_style(key, False)

        self.apply_mesh_control_enabled()
        self.update_title()

    def toggle_internal_lock(self):
        self.internal_editable = bool(self.internal_lock_btn.isChecked())
        self.internal_lock_btn.setText("Lock" if self.internal_editable else "Unlock")
        self.apply_internal_enabled(self.internal_editable)

    def apply_internal_enabled(self, enabled):
        for spec in SCHEMA_ORDER:
            if spec.section != "Internal Information":
                continue
            self.widgets_by_key[spec.key].setEnabled(enabled)

    def apply_mesh_control_enabled(self):
        mc = self.cache.get("mesh_control", None)
        gpu_w = self.widgets_by_key["gpu_memory"]
        cell_w = self.widgets_by_key["cell_size"]

        if mc == "gpu_memory":
            gpu_w.setEnabled(True)
            cell_w.setEnabled(False)
        elif mc == "cell_size":
            gpu_w.setEnabled(False)
            cell_w.setEnabled(True)
        else:
            gpu_w.setEnabled(False)
            cell_w.setEnabled(False)

    def poll_file(self):
        if not self.file_path:
            return
        st = self.stat_file()
        if st is None:
            return
        if self.last_stat is None:
            self.last_stat = st
            return
        if st == self.last_stat:
            return
        self.flash_yellow()
        self.handle_external_change()

    def handle_external_change(self):
        ok, txt_or_err = self.read_text()
        if not ok:
            self.status.showMessage(f"Read failed: {txt_or_err}", 3000)
            return

        disk_values_raw, lines, found = parse_file_text(txt_or_err)
        disk_values = {}
        for spec in SCHEMA_ORDER:
            disk_values[spec.key] = disk_values_raw.get(spec.key, None)

        conflicts = []

        for spec in SCHEMA_ORDER:
            key = spec.key
            old = self.baseline.get(key, None)
            new_disk = disk_values.get(key, None)
            is_dirty = key in self.dirty
            cache_val = self.cache.get(key, None)

            if is_dirty and old != new_disk and cache_val != new_disk:
                conflicts.append(
                    {
                        "key": key,
                        "section": spec.section,
                        "label": spec.label,
                        "cache": "" if cache_val is None else format_value(key, cache_val),
                        "file": "" if new_disk is None else format_value(key, new_disk),
                    }
                )

            if (not is_dirty) and cache_val != new_disk:
                self.cache[key] = new_disk

        self.last_lines = lines
        self.found_keys = found
        self.last_stat = self.stat_file()

        if conflicts:
            dlg = ConflictDialog(self, conflicts)
            res = dlg.exec_()
            if res == QtWidgets.QDialog.Accepted:
                choices = dlg.choices()
                for c in conflicts:
                    key = c["key"]
                    spec = SCHEMA_BY_KEY[key]
                    choose_key = f"{spec.section}::{spec.label}"
                    decision = choices.get(choose_key, "Keep cache")
                    if decision == "Use file":
                        self.cache[key] = disk_values.get(key, None)
                        if key in self.dirty:
                            self.dirty.remove(key)

        self.baseline = dict(disk_values)
        self.sync_ui_from_cache()
        self.status.showMessage("Synced with disk", 2000)

    def validate_before_save(self):
        for spec in SCHEMA_ORDER:
            v = self.cache.get(spec.key, None)
            if v is None:
                continue
            if spec.ftype == "mesh_control" and v not in ("gpu_memory", "cell_size"):
                return False, "Mesh size control is invalid."
            if spec.ftype == "list":
                if (not isinstance(v, list)) or len(v) != spec.list_len:
                    return False, f"{spec.label} is incomplete."
        return True, ""

    def save_to_disk(self):
        if not self.file_path:
            QtWidgets.QMessageBox.information(self, "No file", "Select a file first.")
            return

        ok, msg = self.validate_before_save()
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Invalid input", msg)
            return

        while True:
            ok_write, err = self.try_write_preserve_lines()
            if ok_write:
                self.baseline = dict(self.cache)
                self.dirty.clear()
                self.last_stat = self.stat_file()
                self.sync_ui_from_cache()
                self.status.showMessage("Saved", 2000)
                self.flash_green()
                return

            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("File is busy")
            box.setText("Saving failed. The file may be in use by another program.")
            box.setInformativeText(err)
            retry = box.addButton("Retry", QtWidgets.QMessageBox.AcceptRole)
            cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(retry)
            box.exec_()
            if box.clickedButton() == cancel:
                self.status.showMessage("Save canceled", 2000)
                return

    def try_write_preserve_lines(self):
        folder = os.path.dirname(self.file_path) or "."
        base = os.path.basename(self.file_path)
        tmp_name = f".{base}.tmp.{int(time.time() * 1000)}"
        tmp_path = os.path.join(folder, tmp_name)

        present_in_file = set(self.found_keys)
        updated_lines = []

        for raw in self.last_lines:
            content, comment = strip_inline_comment(raw)
            stripped = content.strip()
            if "=" not in stripped:
                updated_lines.append(raw)
                continue
            left, _right = stripped.split("=", 1)
            key = left.strip()
            if key not in KNOWN_KEYS:
                updated_lines.append(raw)
                continue

            v = self.cache.get(key, None)
            if v is None:
                updated_lines.append(raw)
                continue

            new_text = f"{key} = {format_value(key, v)}"
            if comment:
                new_text = new_text.rstrip("\n").rstrip("\r") + " " + comment.strip()

            if raw.endswith("\r\n"):
                new_text += "\r\n"
            elif raw.endswith("\n"):
                new_text += "\n"
            else:
                new_text += "\n"
            updated_lines.append(new_text)

        appended_any = False
        for spec in SCHEMA_ORDER:
            key = spec.key
            if key in present_in_file:
                continue
            if key not in self.dirty:
                continue
            v = self.cache.get(key, None)
            if v is None:
                continue
            if not appended_any:
                if len(updated_lines) and not updated_lines[operator.sub(len(updated_lines), 1)].endswith("\n"):
                    idx = operator.sub(len(updated_lines), 1)
                    updated_lines[idx] = updated_lines[idx] + "\n"
                updated_lines.append("\n")
                appended_any = True
            updated_lines.append(f"{key} = {format_value(key, v)}\n")

        out_text = "".join(updated_lines)

        try:
            with open(tmp_path, "w", encoding="utf8", newline="") as f:
                f.write(out_text)
            os.replace(tmp_path, self.file_path)

            ok, txt_or_err = self.read_text()
            if ok:
                _vals, lines, found = parse_file_text(txt_or_err)
                self.last_lines = lines
                self.found_keys = found

            return True, ""
        except OSError as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            if isinstance(e, PermissionError) or getattr(e, "errno", None) in (errno.EACCES, errno.EPERM, errno.EBUSY):
                return False, str(e)
            return False, str(e)

    def closeEvent(self, event):
        if self.dirty:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("Unsaved changes")
            box.setText("You have unsaved changes in the cache.")
            box.setInformativeText("Close anyway and discard them?")
            close_btn = box.addButton("Close", QtWidgets.QMessageBox.AcceptRole)
            cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(cancel_btn)
            box.exec_()
            if box.clickedButton() == cancel_btn:
                event.ignore()
                return
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
