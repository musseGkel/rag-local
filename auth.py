from fastapi import Header, HTTPException

API_KEY = "j23m2n4o5p6q7r8s9t0"


def verify_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
