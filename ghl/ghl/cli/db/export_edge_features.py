from typing import Union

from pathlib import Path

from cloup import command, option, option_group, help_option, Path as PathType

from .db_options import db_options


@command(
    name="export_edge_features",
    help="""
        Export all features per edge type from the database. One output file
        will be generated for each edge type.
        
        WARNING: If a edge has multiple types, it will be present in multiple
        files.
    """,
    no_args_is_help=True,
)
@option_group(
    "Export options",
    option(
        "-o",
        "--output_directory",
        type=PathType(file_okay=False, dir_okay=True, writable=True, path_type=Path),
        required=True,
    ),
    option(
        "--edge_types",
        type=str,
        required=False,
        default=None,
        help='List of edge types to export as comma separated list. Example: "A, B"',
    ),
    option(
        "--exclude_regex",
        type=str,
        required=False,
        default=None,
        help="Node types matching the specified prefix are excluded from export.",
    ),
)
@db_options
@help_option("-h", "--help")
def export_edge_features(
    output_directory: Path,
    db_type: str,
    db_uri: str,
    db_user: str,
    db_password: str,
    db_name: Union[None, str],
    edge_types: Union[str, None],
    exclude_regex: Union[str, None],
):
    edge_types_list = (
        None if edge_types is None else [s.strip() for s in edge_types.split(",")]
    )

    if db_type == "neo4j":
        from ghl.dblib.neo4jlib import Neo4JDatabase

        db = Neo4JDatabase(uri=db_uri, auth=(db_user, db_password), database=db_name)
        db.ensure_connectivity()

        output_directory.mkdir(parents=True, exist_ok=True)

        db.export_edge_features(
            output_directory=output_directory,
            edge_types=edge_types_list,
            exclude_regex=exclude_regex,
        )
    else:
        raise NotImplementedError()
