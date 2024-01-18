from cloup import option_group, option, Choice

DB_TYPES = ["neo4j"]

db_options = option_group(
    "Database options",
    option(
        "--db_type",
        type=Choice(DB_TYPES),
        required=True,
        default=DB_TYPES[0],
        envvar="DB_TYPE",
        show_envvar=True,
        help="Type of database.",
    ),
    option(
        "--db_uri",
        type=str,
        required=True,
        envvar="DB_URI",
        show_envvar=True,
        help="Database URI",
    ),
    option(
        "--db_user",
        type=str,
        required=True,
        envvar="DB_USER",
        show_envvar=True,
        help="Database user name",
    ),
    option(
        "--db_password",
        type=str,
        required=True,
        prompt=True,
        hide_input=True,
        envvar="DB_PASSWORD",
        show_envvar=True,
        help="Database user password",
    ),
    option(
        "--db_name",
        type=str,
        required=False,
        default=None,
        envvar="DB_NAME",
        show_envvar=True,
        help="Database name.",
    ),
)
