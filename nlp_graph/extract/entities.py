"""Shared B2B entity / allowlist constants."""

KNOWN_ENTITIES: frozenset[str] = frozenset(
    e.lower()
    for e in """
    databricks snowflake dbt bigquery redshift sagemaker vertex fabric firebolt
    starburst trino looker tableau power bi airflow spark delta lakehouse
    """.split()
)

B2B_ALLOWLIST: frozenset[str] = KNOWN_ENTITIES | frozenset(
    {
        "lakehouse",
        "migration",
        "pricing",
        "serverless",
        "governance",
        "consumption",
        "lock-in",
        "vendor",
        "unity catalog",
        "dbt cloud",
        "dbt core",
    }
)
