import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from src.summarize import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)