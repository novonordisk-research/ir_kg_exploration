from cloup import command, version_option, help_option, group
from .. import __version__


@group(
    name="aNNGEL",
    help="""
        aNNGEL is a library and CLI for facilitating (knowledge) graph embedding. 
    """,
    no_args_is_help=True,
)
@help_option(
    "--help",
    "-h",
)
@version_option(
    __version__,
    "--version",
    "-v",
)
def _main():
    pass


from .db import db

_main.add_command(db)

from .algorithms import algorithms

_main.add_command(algorithms)


def main():
    # needed to be able to pass regex pattern on windows
    _main.main(windows_expand_args=False)
