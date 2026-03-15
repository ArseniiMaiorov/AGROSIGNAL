"""Topology cleanup SQL for PostGIS."""


def get_topology_cleanup_sql(*, simplify_enabled: bool = False, simplify_tol_deg: float = 0.000025) -> list[str]:
    """Return list of SQL statements for topology cleanup.

    Each statement uses :run_id as a named parameter.

    Returns:
        List of SQL strings.
    """
    queries = [
        # 1. Fix invalid geometries
        """
        UPDATE fields
        SET geom = ST_Multi(ST_MakeValid(geom))
        WHERE aoi_run_id = :run_id
          AND NOT ST_IsValid(geom);
        """,
        # 2. Remove overlaps: smaller fields yield to larger
        """
        UPDATE fields f1
        SET geom = ST_Multi(ST_CollectionExtract(
            ST_MakeValid(ST_Difference(f1.geom, (
                SELECT COALESCE(ST_Union(f2.geom), ST_GeomFromText('GEOMETRYCOLLECTION EMPTY', 4326))
                FROM fields f2
                WHERE f2.aoi_run_id = :run_id
                  AND f2.id != f1.id
                  AND f2.area_m2 > f1.area_m2
                  AND ST_Intersects(f1.geom, f2.geom)
            ))), 3))
        WHERE f1.aoi_run_id = :run_id
          AND EXISTS (
              SELECT 1 FROM fields f2
              WHERE f2.aoi_run_id = :run_id
                AND f2.id != f1.id
                AND f2.area_m2 > f1.area_m2
                AND ST_Intersects(f1.geom, f2.geom)
          );
        """,
        # 3. Remove slivers (< 200 m2)
        """
        DELETE FROM fields
        WHERE aoi_run_id = :run_id
          AND ST_Area(geom::geography) < 200;
        """,
        # 4. Recalculate area and perimeter
        """
        UPDATE fields
        SET area_m2 = ST_Area(geom::geography),
            perimeter_m = ST_Perimeter(geom::geography)
        WHERE aoi_run_id = :run_id;
        """,
    ]
    if simplify_enabled:
        queries.insert(
            3,
            f"""
            UPDATE fields f
            SET geom = ST_Multi(
                CASE
                    WHEN (
                        SELECT COALESCE(SUM(ST_NumInteriorRings((dump).geom)), 0)
                        FROM ST_Dump(f.geom) AS dump
                    ) > 0
                    THEN ST_MakeValid(f.geom)
                    ELSE ST_SimplifyPreserveTopology(f.geom, {float(simplify_tol_deg):.10f})
                END
            )
            WHERE f.aoi_run_id = :run_id;
            """,
        )
    return queries
