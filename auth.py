#!/usr/bin/env python3
"""
Authentication and Security Module
Handles JWT tokens, rate limiting, and input validation
"""

import jwt
from jwt import InvalidTokenError
import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Simple user database (use proper database in production)
USERS_DB = {
    "admin": {
        "username": "admin", 
        "password": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # admin123
        "role": "admin"
    },
    "user": {
        "username": "user",
        "password": "e606e38b0d8c19b24cf0ee3808183162ea7cd63ff7912dbb22b5e803286b4446",  # user123
        "role": "user"
    }
}

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class SecurityValidator:
    """Handles security validation and sanitization"""
    
    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"forget\s+everything\s+above",
        r"you\s+are\s+now\s+a",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"human\s*:\s*",
        r"<\s*script\s*>",
        r"javascript\s*:",
        r"eval\s*\(",
        r"exec\s*\(",
        r"import\s+os",
        r"import\s+subprocess",
        r"__import__",
    ]
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input must be a string"
            )
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', text)
        
        # Limit input length
        if len(sanitized) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text too long (max 2000 characters)"
            )
        
        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"Potential injection attempt detected: {pattern}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Input contains potentially harmful content"
                )
        
        return sanitized.strip()
    
    @classmethod
    def validate_generation_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize generation parameters"""
        validated = {}
        
        # Max tokens validation
        max_tokens = params.get('max_tokens', 100)
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_tokens must be an integer between 1 and 500"
            )
        validated['max_tokens'] = max_tokens
        
        # Temperature validation
        temperature = params.get('temperature', 0.7)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="temperature must be a number between 0 and 2"
            )
        validated['temperature'] = float(temperature)
        
        # Top_p validation
        top_p = params.get('top_p', 0.9)
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_p must be a number between 0 and 1"
            )
        validated['top_p'] = float(top_p)
        
        return validated

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = USERS_DB.get(username)
    if not user:
        return None
    
    if not verify_password(password, user["password"]):
        return None
    
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_auth_user(request: Request) -> Optional[Dict[str, Any]]:
    """Simple function to get authenticated user from request"""
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        payload = verify_token(token)
        username = payload.get("sub")
        
        if username and username in USERS_DB:
            return USERS_DB[username]
        return None
    except:
        return None

class JWTBearer(HTTPBearer):
    """Custom JWT Bearer authentication"""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

# Create instance of JWT bearer
jwt_bearer = JWTBearer()

# Rate limiting storage (use Redis in production)
request_counts = {}
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str, username: Optional[str] = None) -> bool:
    """Check if request is within rate limits"""
    now = datetime.utcnow()
    
    # Use username if authenticated, otherwise use IP
    key = username if username else client_ip
    
    if key not in request_counts:
        request_counts[key] = []
    
    # Remove old requests outside the window
    request_counts[key] = [
        req_time for req_time in request_counts[key]
        if (now - req_time).total_seconds() < RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(request_counts[key]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    request_counts[key].append(now)
    return True

async def get_current_user(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = USERS_DB.get(username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user
