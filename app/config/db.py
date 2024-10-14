from sqlalchemy import create_engine, MetaData

engine = create_engine("mysql+pymysql://root:root@db:3306/danistoredb") # local: localhost, remote: db

meta = MetaData()

conn = engine.connect()