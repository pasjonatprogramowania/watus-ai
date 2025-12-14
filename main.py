import uvicorn
from src.main import app
from src import BASE_API_HOST, BASE_API_PORT

if __name__ == "__main__":
    uvicorn.run(app, host=BASE_API_HOST, port=BASE_API_PORT)