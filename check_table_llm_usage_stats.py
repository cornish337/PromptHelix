from sqlalchemy import create_engine, inspect, text
from prompthelix.database import DATABASE_URL, SessionLocal # SessionLocal for potential direct query if needed

def get_table_schema():
    print(f"Connecting to database: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)

    table_name = "llm_usage_statistics"
    table_exists = inspector.has_table(table_name)

    print(f"\nTable '{table_name}' exists: {table_exists}")

    if table_exists:
        print(f"\n--- Schema for table: {table_name} ---")

        # Columns
        columns = inspector.get_columns(table_name)
        print("\nColumns:")
        for column in columns:
            print(f"  Name: {column['name']}, Type: {column['type']}, "
                  f"Nullable: {column['nullable']}, Default: {column.get('default')}")

        # Primary Key
        pk_constraint = inspector.get_pk_constraint(table_name)
        if pk_constraint and pk_constraint['constrained_columns']:
            print(f"\nPrimary Key Columns: {pk_constraint['constrained_columns']}")
        else:
            print("\nPrimary Key Columns: Not defined or not found by inspector.")


        # Foreign Keys
        foreign_keys = inspector.get_foreign_keys(table_name)
        if foreign_keys:
            print("\nForeign Keys:")
            for fk in foreign_keys:
                print(f"  Name: {fk['name']}")
                print(f"    Constrained Columns: {fk['constrained_columns']}")
                print(f"    Referred Schema: {fk['referred_schema']}")
                print(f"    Referred Table: {fk['referred_table']}")
                print(f"    Referred Columns: {fk['referred_columns']}")
        else:
            print("\nForeign Keys: None found.")

        # Unique Constraints
        unique_constraints = inspector.get_unique_constraints(table_name)
        if unique_constraints:
            print("\nUnique Constraints:")
            for constraint in unique_constraints:
                print(f"  Name: {constraint['name']}, Columns: {constraint['column_names']}")
        else:
            print("\nUnique Constraints: None found.")

        # Indexes
        indexes = inspector.get_indexes(table_name)
        if indexes:
            print("\nIndexes:")
            for index in indexes:
                print(f"  Name: {index['name']}, Columns: {index['column_names']}, Unique: {index['unique']}")
        else:
            print("\nIndexes: None found.")

        print(f"\n--- End of schema for table: {table_name} ---")

    # For SQLite, PRAGMA table_info might give slightly different/more direct output for types
    # This is an alternative way to list columns, especially useful for SQLite type affinities
    if engine.name == 'sqlite':
        print(f"\n--- PRAGMA table_info for {table_name} (SQLite specific) ---")
        with SessionLocal() as db:
            try:
                result = db.execute(text(f"PRAGMA table_info({table_name});"))
                pragma_columns = result.fetchall()
                if pragma_columns:
                    print("cid | name | type | notnull | dflt_value | pk")
                    for row in pragma_columns:
                        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]}")
                else:
                    print("PRAGMA table_info returned no data (table might be empty or truly not exist).")
            except Exception as e:
                print(f"Error executing PRAGMA table_info: {e} (This is expected if table does not exist)")


if __name__ == "__main__":
    get_table_schema()
