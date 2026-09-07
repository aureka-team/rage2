from itertools import cycle

from rich.align import Align
from rich.console import Console
from rich.text import Text

RAGE_BANNER = (
    "██████╗  █████╗  ██████╗ ███████╗",
    "██╔══██╗██╔══██╗██╔════╝ ██╔════╝",
    "██████╔╝███████║██║  ███╗█████╗  ",
    "██╔══██╗██╔══██║██║   ██║██╔══╝  ",
    "██║  ██║██║  ██║╚██████╔╝███████╗",
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
)
RAGE_STYLES = (
    "bold bright_magenta",
    "bold magenta",
    "bold bright_cyan",
)

console = Console()


def render_header() -> None:
    banner = Text()
    banner.append("\n\n")

    for line, style in zip(RAGE_BANNER, cycle(RAGE_STYLES), strict=False):
        banner.append(f"{line}\n", style=style)

    banner.append(
        "R A G   E N G I N E".center(len(RAGE_BANNER[0])),
        style="dim bright_magenta",
    )
    console.print(Align.center(banner))


def render_step(label: str, action: str) -> None:
    message = Text()
    message.append("\n┌─[ ", style="dim magenta")
    message.append(f"{label.upper()} ]\n", style="bold white")
    message.append("└──> ", style="dim magenta")
    message.append(f"{action.upper()}...\n", style="dim white")
    console.print(message)


def render_step_detail(label: str, value: object) -> None:
    detail = Text()
    detail.append(" :: ", style="dim magenta")
    detail.append(label.upper(), style="bold white")
    detail.append(" // ", style="dim magenta")
    detail.append(str(value), style="dim white")
    console.print(detail)
