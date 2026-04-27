from __future__ import annotations

import re


def trim(docstring):
    if not docstring:
        return ""


    lines = docstring.expandtabs().splitlines()

    indent = 232323
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))

    trimmed = [lines[0].strip()]
    if indent < 232323:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())

    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    return "\n".join(trimmed)


def env_name_format(str):

    split = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", str)

    split = filter(lambda x: x.upper() != "ENV", split)
    return " ".join(split)
