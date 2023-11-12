from fastapi import FastAPI

from tech import model

model_app = FastAPI(title="Model API")

@model_app.get("/")
def main():
  return {"msg":"Hii !"}

@model_app.post("/model_predict")
def model_predict(sentence : str):
  result = model.perform(sentence)
  return result