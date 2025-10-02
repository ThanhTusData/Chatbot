# src/serving/auth.py
import os
from fastapi import HTTPException, Depends, Header

REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")

def fake_verify_token(token: str):
    # lightweight placeholder - replace with python-jose verification if you want real JWT
    if token == "dev-token":
        return {"sub": "dev-user"}
    raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(authorization: str = Header(None)):
    if not REQUIRE_AUTH:
        return {"sub": "anonymous"}
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing auth header")
    scheme, _, token = authorization.partition(" ")
    return fake_verify_token(token)
