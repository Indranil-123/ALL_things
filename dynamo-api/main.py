import os
import uuid

import boto3                                   
from botocore.exceptions import ClientError    
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TABLE_NAME = os.getenv("DYNAMODB_TABLE", "contacts")     
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")    

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)  

app = FastAPI(title="DynamoDB Demo API")

class ItemIn(BaseModel):
    name: str
    description: str | None = None   


def health():
    return {"status": "ok"}


# ---------- POST: create an item ----------
@app.post("/items", status_code=201)
def create_item(item: ItemIn):
    record = {"id": str(uuid.uuid4()), **item.model_dump(exclude_none=True)}
    try:
        table.put_item(Item=record)   
    except ClientError as e:
        raise HTTPException(status_code=500, detail=e.response["Error"]["Message"])
    return record


@app.get("/items/{item_id}")
def get_item(item_id: str):
    try:
        response = table.get_item(Key={"id": item_id})
    except ClientError as e:
        raise HTTPException(status_code=500, detail=e.response["Error"]["Message"])

    if "Item" not in response:
        raise HTTPException(status_code=404, detail="Item not found")
    return response["Item"]