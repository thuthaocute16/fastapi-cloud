from typing import Union
from fastapi import FastAPI
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.router import router_main as rt

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rt.router)