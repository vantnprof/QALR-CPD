#!/usr/bin/env python3
"""12 — OPTIONAL: rerun the QAOA rows of Tables 4-7 under the uniform non-empty rule (R2.6). Very long.

    python 12_qaoa_table_rows.py --seeds 0-4                 (about 6 h for 5 seeds; 20 seeds is about 24 h)
    python 12_qaoa_table_rows.py --datasets covid --seeds 0-19

Only needed if step 10 shows that the empty mask wins at the table penalties; otherwise the submitted QAOA rows stand.
Output: revision_out/qaoa_tables/runs.csv
"""
from revision.cli import add_datasets, add_qaoa, run


def opts(ap):
    add_datasets(ap)
    add_qaoa(ap)


if __name__ == "__main__":
    run("qaoa-tables", __doc__, opts, outputs=["qaoa_tables/runs.csv"], next_step="python 14_make_tables.py")
