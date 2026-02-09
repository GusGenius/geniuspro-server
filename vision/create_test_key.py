#!/usr/bin/env python3
"""
Create a test API key for vision service testing.

Run this on the server to create a test key directly in Supabase.
"""

import hashlib
import secrets
import sys
import os

# Add parent to path for supabase import
sys.path.insert(0, os.path.expanduser("~/geniuspro-superintelligence"))
from superintelligence.config import load_config

def generate_api_key() -> str:
    """Generate API key in format: gp-<48 hex chars>"""
    bytes = secrets.token_bytes(24)
    hex_str = bytes.hex()
    return f"gp-{hex_str}"

def hash_key(raw_key: str) -> str:
    """SHA-256 hash of API key"""
    return hashlib.sha256(raw_key.encode()).hexdigest()

def get_key_prefix(key: str) -> str:
    """Get display prefix"""
    return key[:7] + "..." + key[-4:]

def main():
    config = load_config()
    
    if not config.supabase_service_key:
        print("ERROR: SUPABASE_SERVICE_KEY not set in environment")
        sys.exit(1)
    
    # Generate key
    plain_key = generate_api_key()
    key_hash = hash_key(plain_key)
    key_prefix = get_key_prefix(plain_key)
    
    print(f"Generated API key: {plain_key}")
    print(f"Key prefix: {key_prefix}")
    print(f"Key hash: {key_hash[:16]}...")
    print()
    print("To insert into Supabase, run this SQL:")
    print()
    print(f"""
INSERT INTO api_keys (user_id, name, profile, key_hash, key_prefix, is_active, rate_limit_rpm, rate_limit_tpm)
VALUES (
    (SELECT id FROM auth.users LIMIT 1),  -- Use first user, or replace with specific user_id
    'Vision Test Key',
    'vision',  -- or 'universal' for all endpoints
    '{key_hash}',
    '{key_prefix}',
    true,
    1000,  -- 1000 requests per minute
    1000000  -- 1M tokens per minute
);
""")
    print()
    print("Or use the Supabase dashboard:")
    print(f"1. Go to Table Editor > api_keys")
    print(f"2. Insert new row with:")
    print(f"   - name: 'Vision Test Key'")
    print(f"   - profile: 'vision'")
    print(f"   - key_hash: '{key_hash}'")
    print(f"   - key_prefix: '{key_prefix}'")
    print(f"   - is_active: true")
    print()
    print(f"YOUR API KEY (save this!): {plain_key}")
    print()
    print("Test with:")
    print(f'curl -H "X-API-Key: {plain_key}" https://api.geniuspro.io/vision/v1/models')

if __name__ == "__main__":
    main()
