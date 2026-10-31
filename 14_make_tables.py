#!/usr/bin/env python3
"""14 — Build the LaTeX tables and the placeholder report from everything that has been run so far.

    python 14_make_tables.py

Output: revision_out/tables/PLACEHOLDERS.md (one line per value the response letter needs, grouped by reviewer
comment; missing steps are marked MISSING) and revision_out/tables/*.tex (drop-in tabular bodies).
Send revision_out/tables/ back for the response letter and the manuscript text.
"""
from revision.make_tables import main

if __name__ == "__main__":
    main()
