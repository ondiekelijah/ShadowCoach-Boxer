#!/usr/bin/env python3
"""
Run the ShadowCoach API server
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("shadowcoach.api.app:app", host="0.0.0.0", port=8000, reload=True)
