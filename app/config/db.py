from sqlalchemy import create_engine, MetaData
from time import sleep

sleep(20)
engine = create_engine("mysql+pymysql://root:root@localhost:3306/danistoredb") # local: localhost, remote: db

meta = MetaData()

conn = engine.connect()