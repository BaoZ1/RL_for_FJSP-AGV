from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/hello")
def hello():
    return "Hello from backend"

if __name__ == "__main__":
    uvicorn.run(app, reload=False)
