"""JWT authentication."""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from zone_guard.api.schemas import LoginRequest, TokenResponse

router = APIRouter()
security = HTTPBearer(auto_error=False)
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

_secret = "CHANGE_ME"
_algo = "HS256"
_admin_user = "admin"
_admin_hash = pwd.hash("admin")


def configure_auth(jwt_secret, admin_username, admin_password):
    global _secret, _admin_user, _admin_hash
    _secret = jwt_secret
    _admin_user = admin_username
    _admin_hash = pwd.hash(admin_password)


async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(security)):
    if creds is None:
        return {"sub": "anonymous", "role": "guest"}
    try:
        return jwt.decode(creds.credentials, _secret, algorithms=[_algo])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    if req.username != _admin_user or not pwd.verify(req.password, _admin_hash):
        raise HTTPException(status_code=401, detail="Bad credentials")
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    token = jwt.encode({"sub": req.username, "role": "admin", "exp": expire}, _secret, algorithm=_algo)
    return TokenResponse(access_token=token)
