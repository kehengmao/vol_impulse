import os
import sys
import threading
import traceback
import faulthandler
from datetime import datetime
import numpy as np
import ctypes
import gc
import weakref

_is_test = False
ULTRA_DEBUG = os.environ.get("CHART_ULTRA_DEBUG", "1") == "1"
QT_DEBUG_MODE = os.environ.get("CHART_QT_DEBUG") == "1"

is_check_extrema_num = False
is_write_crash_log = False
_check_extrema_lock = threading.Lock()

def check_extrema_num(abs_time_arr, extremas):
    if not is_check_extrema_num:
        return

    with _check_extrema_lock:
        if not hasattr(check_extrema_num, 'seen_peaks'):
            check_extrema_num.seen_peaks = set()

        valid_idx = np.where(np.abs(extremas) > 0.5)[0]
        peaks_abs_time = abs_time_arr[valid_idx]

        for t in peaks_abs_time:
            if t > 0:
                check_extrema_num.seen_peaks.add(t)

        print(f"""Status: {len(check_extrema_num.seen_peaks)}""")


def time_it(func):
    return func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_LOG_PATH = os.path.join(BASE_DIR, "chart1_crash.log")

MINIDUMP_DIR = os.path.join(BASE_DIR, "minidumps")
DEBUG_LOG_PATH = os.path.join(BASE_DIR, "chart1_debug.log")


os.environ.setdefault("QT_FORCE_STDERR_LOGGING", "1")
os.environ.setdefault("QT_MESSAGE_PATTERN", "[%{time yyyy-MM-dd HH:mm:ss.zzz}] [%{type}] %{file}:%{line} %{message}")
os.environ.setdefault("QT_FATAL_WARNINGS", "0")

_qt_rules = os.environ.get("CHART_QT_LOGGING_RULES")
if QT_DEBUG_MODE:
    if _qt_rules and isinstance(_qt_rules, str):
        os.environ.setdefault("QT_LOGGING_RULES", _qt_rules)
    os.environ.setdefault("QT_DEBUG_PLUGINS", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=true")

os.environ["QT_MESSAGE_PATTERN"] = "[%{time yyyyMMdd h:mm:ss.zzz} %{if-debug}DEBUG%{endif}%{if-info}INFO%{endif}%{if-warning}WARN%{endif}%{if-critical}CRIT%{endif}%{if-fatal}FATAL%{endif}] %{category}: %{message}"

_log_lock = threading.Lock()
_test_log_lock = threading.Lock()

def write_crash_log(msg):
    if not is_write_crash_log:
        return

    try:
        with _log_lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_line = f"[{timestamp}] {msg}\n"
            with open(CRASH_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())
            if _is_test:
                sys.__stdout__.write(log_line)
                sys.__stdout__.flush()
    except Exception:
        pass

def write_debug_log(msg):
    """Write debug information to separate debug log"""
    if not QT_DEBUG_MODE:
        return
    try:
        with _log_lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_line = f"[{timestamp}] {msg}\n"
            with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(log_line)
                f.flush()
    except Exception:
        pass

class QtMemoryTracker:
    """Track Qt object lifecycle to detect memory issues"""
    def __init__(self):
        self._object_refs = weakref.WeakSet()
        self._creation_stack = {}
        self._lock = threading.Lock()

    def track_object(self, obj, name=""):
        """Track a Qt object for memory debugging"""
        if not QT_DEBUG_MODE:
            return
        try:
            with self._lock:
                self._object_refs.add(obj)
                if name:
                    self._creation_stack[id(obj)] = {
                        'name': name,
                        'stack': traceback.format_stack()[-3:],
                        'created_at': datetime.now()
                    }
        except Exception as e:
            write_debug_log(f"Failed to track object {name}: {e}")

    def validate_object(self, obj, operation=""):
        """Validate Qt object before operations"""
        if not QT_DEBUG_MODE or obj is None:
            return True
        try:
            # Check if object is valid
            if hasattr(obj, 'isVisible'):
                try:
                    obj.isVisible()
                    return True
                except RuntimeError as e:
                    write_crash_log(f"Qt object validation failed during {operation}: {e}")
                    return False
            return True
        except Exception as e:
            write_crash_log(f"Qt object validation error during {operation}: {e}")
            return False

    def get_stats(self):
        """Get memory tracking statistics"""
        if not QT_DEBUG_MODE:
            return "Qt debug mode disabled"
        try:
            with self._lock:
                alive_count = len(self._object_refs)
                return f"Qt objects tracked: {alive_count}"
        except Exception:
            return "Failed to get stats"

# Global Qt memory tracker
qt_memory_tracker = QtMemoryTracker()

def safe_qt_operation(operation_name, func, *args, **kwargs):
    """Safely execute Qt operations with error catching"""
    if not QT_DEBUG_MODE:
        return func(*args, **kwargs)

    try:
        write_debug_log(f"Qt operation: {operation_name}")
        result = func(*args, **kwargs)
        write_debug_log(f"Qt operation success: {operation_name}")
        return result
    except RuntimeError as e:
        write_crash_log(f"Qt RuntimeError in {operation_name}: {e}")
        write_debug_log(f"Qt RuntimeError details: {traceback.format_exc()}")
        raise
    except Exception as e:
        write_crash_log(f"Qt Exception in {operation_name}: {e}")
        write_debug_log(f"Qt Exception details: {traceback.format_exc()}")
        raise

def validate_qt_pointer(obj, obj_name="QtObject"):
    """Validate Qt pointer before use"""
    if obj is None:
        return False
    if not QT_DEBUG_MODE:
        return True
    try:
        # Try to access a safe property to validate object
        if hasattr(obj, 'parent'):
            obj.parent()  # This will raise RuntimeError if object is deleted
            return True
        return True
    except RuntimeError:
        write_crash_log(f"Qt pointer validation failed for {obj_name}: object likely deleted")
        return False
    except Exception as e:
        write_crash_log(f"Qt pointer validation error for {obj_name}: {e}")
        return False

if ULTRA_DEBUG:
    try:
        _fault_file = open(CRASH_LOG_PATH, 'a', encoding='utf-8')
        faulthandler.enable(_fault_file)
        if hasattr(faulthandler, 'register'):
            import signal
            for sig in (signal.SIGABRT, signal.SIGILL, signal.SIGSEGV, signal.SIGFPE):
                try: faulthandler.register(sig, file=_fault_file)
                except Exception: pass
    except Exception:
        pass

    try:
        write_crash_log(f"[BOOT] QT_FORCE_STDERR_LOGGING={os.environ.get('QT_FORCE_STDERR_LOGGING')}")
        write_crash_log(f"[BOOT] QT_MESSAGE_PATTERN={os.environ.get('QT_MESSAGE_PATTERN')}")
        write_crash_log(f"[BOOT] QT_LOGGING_RULES={os.environ.get('QT_LOGGING_RULES')}")
        write_crash_log(f"[BOOT] MINIDUMP_DIR={MINIDUMP_DIR}")
    except Exception:
        pass

    def thread_exception_handler(args):
        try:
            error_msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:
            error_msg = f"Failed to format exception: {args.exc_value}"
        t_name = args.thread.name if args.thread else "Unknown Thread"
        write_crash_log(f"================= THREAD FATAL EXCEPTION ({t_name}) =================")
        write_crash_log(error_msg)
        write_crash_log("==========================================================")
    threading.excepthook = thread_exception_handler

    def unraisable_handler(unraisable):
        try:
            error_msg = "".join(traceback.format_exception(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback))
        except Exception:
            error_msg = f"Failed to format exception: {unraisable.exc_value}"
        write_crash_log(f"================= UNRAISABLE EXCEPTION ({unraisable.err_msg}) =================")
        write_crash_log(error_msg)
        write_crash_log("==========================================================")
    sys.unraisablehook = unraisable_handler

    def global_exception_handler(exctype, value, tb):
        try:
            error_msg = "".join(traceback.format_exception(exctype, value, tb))
        except Exception:
            error_msg = f"Failed to format exception: {value}"
        write_crash_log("================= UNCAUGHT FATAL EXCEPTION =================")
        write_crash_log(error_msg)
        write_crash_log("==========================================================")
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = global_exception_handler

    from PySide6.QtCore import qInstallMessageHandler, QtMsgType, QThread, QObject
    from PySide6.QtWidgets import QApplication

    def qt_message_handler(msg_type, context, msg):
        type_str = "UNKNOWN"
        if msg_type == QtMsgType.QtDebugMsg: type_str = "DEBUG"
        elif msg_type == QtMsgType.QtInfoMsg: type_str = "INFO"
        elif msg_type == QtMsgType.QtWarningMsg: type_str = "WARNING"
        elif msg_type == QtMsgType.QtCriticalMsg: type_str = "CRITICAL"
        elif msg_type == QtMsgType.QtFatalMsg: type_str = "FATAL"

        current_thread = QThread.currentThread()
        thread_info = f"Thread: {current_thread.objectName() if current_thread.objectName() else hex(int(current_thread.currentThreadId()))}"

        log_msg = f"[QT {type_str}] {thread_info} {context.file}:{context.line} - {msg}"

        if msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg, QtMsgType.QtWarningMsg):
            write_crash_log(log_msg)
            if QT_DEBUG_MODE:
                write_debug_log(log_msg)
            if msg_type == QtMsgType.QtFatalMsg:
                py_stack = "".join(traceback.format_stack())
                write_crash_log("Python Stack at Qt Fatal:\n" + py_stack)

                # Additional memory debugging for fatal errors
                if QT_DEBUG_MODE:
                    write_debug_log(f"Qt Memory Stats: {qt_memory_tracker.get_stats()}")
                    write_debug_log(f"GC Stats: {gc.get_stats()}")
                    write_debug_log(f"Active Threads: {threading.active_count()}")

        if QT_DEBUG_MODE and msg_type in (QtMsgType.QtDebugMsg, QtMsgType.QtInfoMsg):
            write_debug_log(log_msg)

    qInstallMessageHandler(qt_message_handler)

    class StderrInterceptor:
        def write(self, msg):
            if msg and isinstance(msg, str) and msg.strip():
                write_crash_log(f"[STDERR] {msg.strip()}")
            sys.__stderr__.write(msg)
        def flush(self):
            sys.__stderr__.flush()
    sys.stderr = StderrInterceptor()

    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        STATUS_ACCESS_VIOLATION = 0xC0000005
        STATUS_STACK_OVERFLOW = 0xC00000FD
        STATUS_FLOAT_DIVIDE_BY_ZERO = 0xC000008E
        STATUS_HEAP_CORRUPTION = 0xC0000374

        def _ensure_minidump_dir():
            try:
                os.makedirs(MINIDUMP_DIR, exist_ok=True)
            except Exception:
                pass

        def _write_minidump(exception_pointers):
            try:
                _ensure_minidump_dir()
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                dump_path = os.path.join(MINIDUMP_DIR, f"chart_crash_{ts}.dmp")

                GENERIC_WRITE = 0x40000000
                FILE_SHARE_READ = 0x00000001
                FILE_SHARE_WRITE = 0x00000002
                CREATE_ALWAYS = 2
                FILE_ATTRIBUTE_NORMAL = 0x00000080

                h_file = ctypes.windll.kernel32.CreateFileW(
                    dump_path,
                    GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    None,
                    CREATE_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL,
                    None,
                )
                invalid_handle = ctypes.c_void_p(-1).value
                if (h_file is None) or (h_file == 0) or (h_file == invalid_handle):
                    err = ctypes.windll.kernel32.GetLastError()
                    write_crash_log(f"[MINIDUMP] CreateFileW failed, err={err}")
                    return None

                class MINIDUMP_EXCEPTION_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("ThreadId", wintypes.DWORD),
                        ("ExceptionPointers", ctypes.c_void_p),
                        ("ClientPointers", wintypes.BOOL),
                    ]

                mini_exc = MINIDUMP_EXCEPTION_INFORMATION()
                mini_exc.ThreadId = ctypes.windll.kernel32.GetCurrentThreadId()
                mini_exc.ExceptionPointers = ctypes.cast(exception_pointers, ctypes.c_void_p).value if exception_pointers else 0
                mini_exc.ClientPointers = False

                MiniDumpWriteDump = ctypes.windll.dbghelp.MiniDumpWriteDump
                MiniDumpWriteDump.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    ctypes.POINTER(MINIDUMP_EXCEPTION_INFORMATION),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                MiniDumpWriteDump.restype = wintypes.BOOL

                MiniDumpWithIndirectlyReferencedMemory = 0x00000040
                MiniDumpScanMemory = 0x00000010
                MiniDumpWithUnloadedModules = 0x00000020
                MiniDumpWithThreadInfo = 0x00001000
                dump_type = (
                    MiniDumpWithIndirectlyReferencedMemory
                    | MiniDumpScanMemory
                    | MiniDumpWithUnloadedModules
                    | MiniDumpWithThreadInfo
                )

                h_proc = ctypes.windll.kernel32.GetCurrentProcess()
                pid = ctypes.windll.kernel32.GetCurrentProcessId()

                ok = MiniDumpWriteDump(
                    h_proc,
                    pid,
                    h_file,
                    dump_type,
                    ctypes.byref(mini_exc) if mini_exc.ExceptionPointers else None,
                    None,
                    None,
                )
                ctypes.windll.kernel32.CloseHandle(h_file)

                if not ok:
                    err = ctypes.windll.kernel32.GetLastError()
                    write_crash_log(f"[MINIDUMP] MiniDumpWriteDump failed, err={err}")
                    return None

                write_crash_log(f"[MINIDUMP] saved: {dump_path}")
                return dump_path
            except Exception as e:
                write_crash_log(f"[MINIDUMP] exception: {e}")
                return None
        STATUS_HEAP_CORRUPTION = 0xC0000374

        class EXCEPTION_RECORD(ctypes.Structure):
            pass
        EXCEPTION_RECORD._fields_ = [
            ("ExceptionCode", wintypes.DWORD),
            ("ExceptionFlags", wintypes.DWORD),
            ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
            ("ExceptionAddress", wintypes.LPVOID),
            ("NumberParameters", wintypes.DWORD),
            ("ExceptionInformation", wintypes.ULONG * 15),
        ]

        class EXCEPTION_POINTERS(ctypes.Structure):
            _fields_ = [
                ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
                ("ContextRecord", wintypes.LPVOID),
            ]

        TOP_LEVEL_EXCEPTION_FILTER = ctypes.WINFUNCTYPE(
            wintypes.LONG,
            ctypes.POINTER(EXCEPTION_POINTERS)
        )

        def windows_seh_handler(exception_pointers):
            try:
                record = exception_pointers.contents.ExceptionRecord.contents
                code = record.ExceptionCode
                addr = record.ExceptionAddress

                reason = "Unknown C++ Exception"
                if code == STATUS_ACCESS_VIOLATION:
                    reason = "Access Violation"
                elif code == STATUS_STACK_OVERFLOW:
                    reason = "Stack Overflow"
                elif code == STATUS_FLOAT_DIVIDE_BY_ZERO:
                    reason = "Float Divide By Zero"
                elif code == STATUS_HEAP_CORRUPTION:
                    reason = "Heap Corruption (0xc0000374)"

                error_msg = f"================= WINDOWS SEH FATAL CRASH =================\n"
                error_msg += f"Exception Code: 0x{code:08X}\n"
                error_msg += f"Faulting Address: {hex(addr) if addr else 'NULL'}\n"
                error_msg += f"Reason: {reason}\n"

                if QT_DEBUG_MODE:
                    error_msg += f"Qt Objects Alive: {qt_memory_tracker.get_stats()}\n"
                    error_msg += f"Python Thread Count: {threading.active_count()}\n"
                    error_msg += f"GC Garbage Count: {len(gc.garbage)}\n"

                error_msg += f"==========================================================="

                write_crash_log(error_msg)

                _write_minidump(exception_pointers)

                py_stack = "".join(traceback.format_stack())
                write_crash_log("Python Stack at Crash:\n" + py_stack)

                if QT_DEBUG_MODE:
                    write_debug_log(f"SEH Exception Details:")
                    write_debug_log(f"Code: 0x{code:08X}")
                    write_debug_log(f"Address: {hex(addr) if addr else 'NULL'}")
                    write_debug_log(f"Reason: {reason}")
                    write_debug_log(f"Thread Info: {threading.current_thread()}")
                    write_debug_log(f"Memory Info: {gc.get_stats()}")

            except Exception as e:
                write_crash_log(f"SEH handler error: {e}")

            return 0

        _seh_handler_ref = TOP_LEVEL_EXCEPTION_FILTER(windows_seh_handler)
        ctypes.windll.kernel32.SetUnhandledExceptionFilter(_seh_handler_ref)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

class QtDebugContext:
    """Context manager for Qt debugging operations"""
    def __init__(self, operation_name, validate_objects=None):
        self.operation_name = operation_name
        self.validate_objects = validate_objects or []
        self.start_time = None

    def __enter__(self):
        if not QT_DEBUG_MODE:
            return self
        self.start_time = datetime.now()
        write_debug_log(f"Qt operation started: {self.operation_name}")

        # Validate objects before operation
        for obj, name in self.validate_objects:
            if not qt_memory_tracker.validate_object(obj, f"{self.operation_name} pre-check"):
                write_crash_log(f"Qt object validation failed for {name} before {self.operation_name}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not QT_DEBUG_MODE:
            return

        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        if exc_type:
            write_crash_log(f"Qt operation failed: {self.operation_name} after {duration:.3f}s")
            write_debug_log(f"Qt operation exception: {self.operation_name} - {exc_type.__name__}: {exc_val}")
            write_debug_log(f"Traceback: {traceback.format_tb(exc_tb)}")
        else:
            write_debug_log(f"Qt operation completed: {self.operation_name} in {duration:.3f}s")

def qt_object_lifetime_tracker(obj_class):
    """Decorator to track Qt object lifecycle"""
    if not QT_DEBUG_MODE:
        return obj_class

    original_init = obj_class.__init__
    original_del = getattr(obj_class, '__del__', None)

    def tracked_init(self, *args, **kwargs):
        try:
            original_init(self, *args, **kwargs)
            qt_memory_tracker.track_object(self, obj_class.__name__)
            write_debug_log(f"Qt object created: {obj_class.__name__} at {hex(id(self))}")
        except Exception as e:
            write_crash_log(f"Qt object creation failed for {obj_class.__name__}: {e}")
            raise

    def tracked_del(self):
        try:
            write_debug_log(f"Qt object destroyed: {obj_class.__name__} at {hex(id(self))}")
            if original_del:
                original_del(self)
        except Exception as e:
            write_crash_log(f"Qt object destruction error for {obj_class.__name__}: {e}")

    obj_class.__init__ = tracked_init
    obj_class.__del__ = tracked_del

    return obj_class

def test_log(msg):
    if _is_test or QT_DEBUG_MODE:
        write_debug_log(f"[TEST] {msg}")
