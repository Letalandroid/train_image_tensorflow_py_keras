from fastapi import APIRouter, HTTPException
from model.predict import predict as predict_table
from config.db import engine

data = APIRouter()

@data.get("/getAll")
async def getAll():
    with engine.connect() as conn:
        try:
            result = conn.execute(predict_table.select()).fetchall()
            return [
                {
                    "fecha": row.fecha,
                    "clothe_id": row.clothe_id,
                    "tipo": row.tipo,
                    "image": row.image,
                    "porcent": row.porcent,
                    "time_calc": row.time_calc
                 } for row in result]
        except Exception as e:
            print(f"Error occurred: {e}")
            raise HTTPException(status_code=500, detail=str(e))

