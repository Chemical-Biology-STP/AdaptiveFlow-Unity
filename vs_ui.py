"""Terminal helpers for the interactive assistant: coloured messages, prompts that re-ask until the answer is valid,
tables and Tab completion.

Adapted from dock_cli.py of the anchored_docking_workflow repository (MIT License, see NOTICE.md).
"""

import glob
import math
import os
import re
import readline
import sys

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
WARNINGS = []     # recorded warnings, repeated on the review screen


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(text):
    return _c("1", text)


def dim(text):
    return _c("2", text)


def red(text):
    return _c("31", text)


def yellow(text):
    return _c("33", text)


def say(msg=""):
    print(msg)


def ok(msg):
    print(f"  {_c('32', '[ok]')} {msg}")


def warn(msg, record=True):
    """Print a warning. Recorded warnings are repeated in the final review."""
    if record:
        WARNINGS.append(msg)
    print(f"  {_c('33', '[!]')}  {msg}")


def fail(msg):
    print(f"  {_c('31', '[x]')}  {msg}")


def note(msg):
    print(f"       {dim(msg)}")


def heading(step, total, title):
    text = f"Step {step} of {total}: {title}"
    print(f"\n{bold(text)}\n{'-' * len(text)}")


def table(rows, headers=None, indent="  "):
    rows = [[str(c) for c in r] for r in rows]
    if headers:
        rows = [list(headers)] + rows
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for n, r in enumerate(rows):
        print(indent + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if headers and n == 0:
            print(indent + "  ".join("-" * w for w in widths))


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def _set_completer(func):
    readline.set_completer(func)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def path_completer(text, state):
    text = os.path.expanduser(text)
    matches = glob.glob(text + "*")
    matches = [m + "/" if os.path.isdir(m) else m for m in matches]
    try:
        return sorted(matches)[state]
    except IndexError:
        return None


def choice_completer(choices):
    def completer(text, state):
        matches = [c for c in choices if c.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None

    return completer


def ask(label, default=None, validate=None, completer=None, required=True):
    """Prompt until the answer is valid. validate() may raise ValueError."""
    _set_completer(completer)
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        value = raw if raw else default
        if value in (None, ""):
            if required:
                print("       Please enter a value.")
                continue
            return None
        if validate is not None:
            try:
                value = validate(value)
            except ValueError as exc:
                fail(str(exc))
                continue
        return value


def ask_yes_no(label, default=False):
    def validate(text):
        t = str(text).lower()
        if t in ("y", "yes", "true"):
            return True
        if t in ("n", "no", "false"):
            return False
        raise ValueError("Please answer yes or no.")

    return ask(f"{label} (yes/no)", "yes" if default else "no", validate, choice_completer(["yes", "no"]))


def ask_number(label, default, cast=float, minimum=None, required=True, maximum=None):
    def validate(text):
        try:
            value = cast(text)
        except (TypeError, ValueError):
            raise ValueError("Please enter a number.")
        if minimum is not None and value < minimum:
            raise ValueError(f"Please enter a number of at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Please enter a number of at most {maximum}.")
        return value

    return ask(label, None if default is None else str(default), validate, required=required)


def ask_menu(options, default=1):
    """Ask for one of several numbered options [(key, text), ...]; returns the key."""
    for n, (_, text) in enumerate(options, 1):
        say(f"    {n}. {text}")

    def pick(text):
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1][0]
        raise ValueError(f"Please enter a number from 1 to {len(options)}.")

    return ask("Choose a number", str(default), pick)


def file_validator(extensions=()):
    def validate(text):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(text.strip("'\""))))
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")
        if extensions and not path.lower().endswith(tuple(extensions)):
            raise ValueError(f"Expected a {' or '.join(extensions)} file.")
        if os.path.getsize(path) == 0:
            raise ValueError("That file is empty.")
        return path

    return validate


TIME_PATTERN = r"(\d+-)?\d+(:\d{2}){0,2}"


def time_validator(text):
    if not re.fullmatch(TIME_PATTERN, text):
        raise ValueError("Use hours:minutes:seconds, for example 2:00:00 (or days-hours:minutes:seconds).")
    return text


def fmt_minutes(minutes):
    minutes = int(math.ceil(minutes))
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00" if days else f"{hours}:{mins:02d}:00"


def time_to_minutes(text):
    """SLURM time text (minutes, h:m, h:m:s, d-h, d-h:m, d-h:m:s) as minutes."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(p) for p in text.split(":")]
        hours, minutes, seconds = (parts + [0, 0])[:3]
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:        # SLURM reads two numbers as minutes:seconds
            return parts[0] + parts[1] / 60
        hours, minutes, seconds = parts
    return days * 1440 + hours * 60 + minutes + seconds / 60
