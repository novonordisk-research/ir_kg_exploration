from typing import Union

from pathlib import Path

from cloup import command, option, option_group, help_option, Path as PathType

from .db_options import db_options


@command(
    name="export_nodes",
    help="Export node IDs from the database.",
    no_args_is_help=True,
)
@option_group(
    "Export options",
    option(
        "-o",
        "--output_file",
        type=PathType(file_okay=True, dir_okay=False, writable=True, path_type=Path),
        required=True,
    ),
    option(
        "--node_types",
        type=str,
        required=False,
        default=None,
        help='List of node types to export as comma separated list. Example: "A, B"',
    ),
)
@db_options
@option_group(
    "Neo4J options",
    option(
        "--neo4j_properties",
        type=str,
        default=None,
        required=False,
        help="""
            Dictionary describing the properties to be exported from Neo4J. The key is
            property name to be assigned in the output. The value is the CYPHER string
            that is required to produce the output from a selected node "n".
            Examples: '{"labels": "labels(n)"}', '{"my_feature": "n.x"}'

            Note: strings must be escaped on Windows, i.e., '{\\"labels\\": \\"labels(n)\\"}'
        """,
    ),
)
@help_option("-h", "--help")
def export_nodes(
    output_file: Path,
    db_type: str,
    db_uri: str,
    db_user: str,
    db_password: str,
    db_name: Union[None, str],
    node_types: Union[str, None],
    neo4j_properties: Union[str, None],
):
    node_types_list = (
        None if node_types is None else [s.strip() for s in node_types.split(",")]
    )

    if db_type == "neo4j":
        from anngel.dblib.neo4jlib import Neo4JDatabase

        db = Neo4JDatabase(uri=db_uri, auth=(db_user, db_password), database=db_name)
        db.ensure_connectivity()

        if neo4j_properties is not None:
            import json

            properties = json.loads(neo4j_properties)
        else:
            properties = None

        db.export_nodes(
            output_file=output_file,
            node_labels=node_types_list,
            properties=properties,
        )
    else:
        raise NotImplementedError()
