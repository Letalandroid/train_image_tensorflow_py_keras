from sqlalchemy import Column, Table
from sqlalchemy.sql.sqltypes import Integer, String, DateTime
from config.db import meta, engine

predict = Table(
    "clothes",
    meta,
    Column("clothe_id", Integer, primary_key=True),
    Column("fecha", DateTime),
    Column(
        "tipo",
        String(255),
    ),
    Column("image", String(255)),
    Column("porcent", String(10)),
    Column("time_calc", String(10))
)

meta.create_all(engine)