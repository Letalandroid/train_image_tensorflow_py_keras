from sqlalchemy import create_engine, MetaData

engine = create_engine("mysql+pymysql://root:root@localhost:5000/danistoredb")

meta = MetaData()

conn = engine.connect()