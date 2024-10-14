from sqlalchemy import Column, Table
from sqlalchemy.sql.sqltypes import Integer, String
from config.db import meta, engine

predict = Table(
    "clothes",
    meta,
    Column("clothe_id", Integer, primary_key=True),
    Column(
        "tipo",
        String(255),
    ),
    Column("image", String(255))
)

meta.create_all(engine)