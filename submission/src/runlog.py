"""
runlog -- дублирование вывода ноутбука в текстовый файл.

Зачем: удобно отдать один файл целиком вместо копирования выводов по ячейкам.

Что пишется в лог:
  * исходник каждой выполненной ячейки;
  * всё, что ушло в stdout/stderr (print, tqdm, warnings);
  * репр значения последнего выражения ячейки и вызовов display();
  * пометки о построенных фигурах (с опциональным сохранением png);
  * traceback, если ячейка упала.

Что НЕ пишется целиком: слишком длинные строки и слишком длинные выводы
обрезаются (см. LIMITS) -- иначе один `sorted(os.listdir())` на 4000 файлов
превращает лог в бесполезную простыню.

Использование в ноутбуке:

    from runlog import start_log, log_note, finish_log
    start_log("logs/00_diagnostics.md")     # одна строка в первой ячейке
    ...
    finish_log()                            # опционально, в конце

Дальше просто работаете как обычно.
"""
from __future__ import annotations

import datetime as _dt
import io
import re
import sys
import traceback
from pathlib import Path

# ------------------------------------------------------------------- ограничения
LIMITS = dict(
    max_line_chars=300,      # длина одной строки
    max_lines=250,           # строк на ячейку
    max_cell_chars=12000,    # символов на ячейку
    max_src_lines=120,       # строк исходника ячейки
)

_state = dict(path=None, tee_out=None, tee_err=None, hook=None,
              orig_out=None, orig_err=None, orig_display=None,
              figdir=None, save_figs=False, n=0)


# ------------------------------------------------------------------------ утилиты
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _collapse_cr(text: str) -> str:
    """Схлопывает перерисовку строк через \\r (прогресс-бары tqdm) до финального состояния."""
    out = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        out.append(line)
    return "\n".join(out)


def _truncate(text: str) -> str:
    text = _strip_ansi(_collapse_cr(text))

    lines = []
    for ln in text.split("\n"):
        if len(ln) > LIMITS["max_line_chars"]:
            cut = LIMITS["max_line_chars"]
            ln = f"{ln[:cut]}  … [строка обрезана, всего {len(ln)} символов]"
        lines.append(ln)

    if len(lines) > LIMITS["max_lines"]:
        keep = LIMITS["max_lines"] // 2
        lines = (lines[:keep]
                 + [f"… [пропущено {len(lines) - 2 * keep} строк] …"]
                 + lines[-keep:])

    text = "\n".join(lines)
    if len(text) > LIMITS["max_cell_chars"]:
        half = LIMITS["max_cell_chars"] // 2
        text = (text[:half]
                + f"\n… [обрезано {len(text) - 2 * half} символов] …\n"
                + text[-half:])
    return text


class _Tee(io.TextIOBase):
    """Пишет и в исходный поток, и в буфер текущей ячейки."""

    def __init__(self, original):
        self.original = original
        self.buf = io.StringIO()

    def write(self, s):
        self.original.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def take(self) -> str:
        v = self.buf.getvalue()
        self.buf = io.StringIO()
        return v


def _write(text: str):
    p = _state["path"]
    if p is None:
        return
    with open(p, "a", encoding="utf-8") as f:
        f.write(text)


def _short_repr(obj, limit=4000) -> str:
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.max_rows", 40, "display.max_columns", 30,
                                   "display.width", 200):
                r = repr(obj)
        else:
            r = repr(obj)
    except Exception:
        try:
            r = repr(obj)
        except Exception:
            r = f"<{type(obj).__name__}: repr недоступен>"
    return r if len(r) <= limit else r[:limit] + f"  … [обрезано, всего {len(r)}]"


# ---------------------------------------------------------------------- основное
def start_log(path="logs/run.md", save_figs: bool = False, echo: bool = True):
    """
    Включает дублирование вывода в файл `path`.

    save_figs=True дополнительно сохраняет каждую фигуру matplotlib
    в подпапку рядом с логом (полезно, если нужны и картинки).
    """
    if _state["hook"] is not None:
        stop_log()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _state["path"] = path
    _state["n"] = 0
    _state["save_figs"] = save_figs
    _state["figdir"] = path.parent / (path.stem + "_figs")
    if save_figs:
        _state["figdir"].mkdir(exist_ok=True)

    header = (f"# Лог выполнения\n\n"
              f"- файл: `{path}`\n"
              f"- начат: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"- python: {sys.version.split()[0]}\n\n---\n")
    path.write_text(header, encoding="utf-8")

    _state["orig_out"], _state["orig_err"] = sys.stdout, sys.stderr
    _state["tee_out"] = _Tee(sys.stdout)
    _state["tee_err"] = _Tee(sys.stderr)
    sys.stdout, sys.stderr = _state["tee_out"], _state["tee_err"]

    ip = _get_ipython()
    if ip is None:
        print("[runlog] IPython не найден: пишется только stdout/stderr")
    else:
        _patch_display()
        ip.events.register("post_run_cell", _post_run_cell)
        _state["hook"] = _post_run_cell

    if echo:
        print(f"[runlog] пишу в {path.resolve()}")
    return path


def stop_log():
    """Выключает дублирование."""
    ip = _get_ipython()
    if ip is not None and _state["hook"] is not None:
        try:
            ip.events.unregister("post_run_cell", _state["hook"])
        except Exception:
            pass
    _state["hook"] = None
    _unpatch_display()
    if _state["orig_out"] is not None:
        sys.stdout, sys.stderr = _state["orig_out"], _state["orig_err"]
    _state["orig_out"] = _state["orig_err"] = None


def finish_log(note: str = ""):
    """Дописывает подвал и выключает логирование."""
    _write(f"\n---\n\n_Завершено {_dt.datetime.now():%Y-%m-%d %H:%M:%S}_\n")
    if note:
        _write(f"\n{note}\n")
    p = _state["path"]
    stop_log()
    print(f"[runlog] готово: {p}")
    return p


def log_note(text: str):
    """Вписывает произвольный комментарий в лог."""
    _write(f"\n> **Заметка.** {text}\n")


# ------------------------------------------------------------- внутренняя кухня
def _get_ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:
        return None


_displayed: list = []


def _patch_display():
    try:
        import IPython.display as D
    except Exception:
        return
    if _state["orig_display"] is not None:
        return
    _state["orig_display"] = D.display

    def _display(*objs, **kw):
        for o in objs:
            _displayed.append(o)
        # Ссылку на эту обёртку успевают захватить сторонние модули
        # (matplotlib_inline.backend_inline делает это при импорте), и после
        # stop_log она продолжает вызываться. Поэтому при выключенном
        # логировании обёртка не падает, а зовёт настоящий display.
        fn = _state["orig_display"]
        if fn is None:
            import IPython.display as _D
            fn = _D.display
        return fn(*objs, **kw)

    D.display = _display
    import builtins
    builtins.display = _display


def _unpatch_display():
    if _state["orig_display"] is None:
        return
    try:
        import IPython.display as D
        D.display = _state["orig_display"]
        import builtins
        builtins.display = _state["orig_display"]
        # backend_inline держит свою ссылку с момента импорта -- вернуть
        # настоящий display надо и там, иначе фигуры перестают отображаться.
        try:
            import matplotlib_inline.backend_inline as _bi
            _bi.display = _state["orig_display"]
        except Exception:
            pass
    except Exception:
        pass
    _state["orig_display"] = None


_seen_figs: set = set()


def _figures() -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    notes = []
    alive = set()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        alive.add(id(fig))
        if id(fig) in _seen_figs:
            continue          # уже отмечена в предыдущей ячейке (backend не закрыл её)
        _seen_figs.add(id(fig))
        w, h = fig.get_size_inches()
        n_ax = len(fig.axes)
        tag = f"фигура {w:.1f}x{h:.1f} дюйма, осей: {n_ax}"
        if _state["save_figs"]:
            fp = _state["figdir"] / f"cell{_state['n']:03d}_fig{num}.png"
            try:
                fig.savefig(fp, dpi=100, bbox_inches="tight")
                tag += f", сохранена в {fp.name}"
            except Exception as e:
                tag += f", сохранить не удалось: {e}"
        notes.append(tag)
    _seen_figs.intersection_update(alive)
    return notes


def _post_run_cell(result):
    _state["n"] += 1
    n = _state["n"]

    src = ""
    try:
        src = (result.info.raw_cell or "").rstrip()
    except Exception:
        pass
    if src.strip().startswith("%") or "runlog" in src and "start_log" in src:
        pass  # магии и запуск логгера всё равно пишем, просто не фильтруем

    src_lines = src.split("\n")
    if len(src_lines) > LIMITS["max_src_lines"]:
        src = "\n".join(src_lines[:LIMITS["max_src_lines"]]
                        + [f"# … [ещё {len(src_lines) - LIMITS['max_src_lines']} строк]"])

    out = _state["tee_out"].take() if _state["tee_out"] else ""
    err = _state["tee_err"].take() if _state["tee_err"] else ""

    chunks = [f"\n## Ячейка [{n}]\n\n```python\n{src}\n```\n"]

    body = out
    if err.strip():
        body += ("\n" if body else "") + "--- stderr ---\n" + err
    if body.strip():
        chunks.append(f"\n**Вывод:**\n\n```\n{_truncate(body).rstrip()}\n```\n")

    for o in _displayed:
        chunks.append(f"\n**display:**\n\n```\n{_truncate(_short_repr(o))}\n```\n")
    _displayed.clear()

    res = getattr(result, "result", None)
    if res is not None:
        chunks.append(f"\n**Значение:**\n\n```\n{_truncate(_short_repr(res))}\n```\n")

    for tag in _figures():
        chunks.append(f"\n_[{tag}]_\n")

    exc = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        chunks.append(f"\n**ОШИБКА:**\n\n```\n{_truncate(tb).rstrip()}\n```\n")

    _write("".join(chunks))
