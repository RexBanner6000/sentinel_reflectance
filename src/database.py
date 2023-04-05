import psycopg2


def connect_to_db():
    return psycopg2.connect(
        "dbname='sentinel' user='postgres' host='localhost' password='password'"
    )


def check_for_samples_in_db(product_uri: str):
    conn = connect_to_db()
    cur = conn.cursor()
    sql = _generate_check_sql(product_uri)
    cur.execute(sql)

    results = cur.fetchall()
    return True if results[0][0] > 0 else False


def _generate_check_sql(product_uri: str):
    return f"SELECT COUNT(sa.uuid) FROM sentinel2a sa where sa.product_uri = '{product_uri}'"
