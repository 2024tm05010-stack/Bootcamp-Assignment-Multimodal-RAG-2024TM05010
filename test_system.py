#!/usr/bin/env python3
"""
Test script for the Multimodal RAG system.
This script tests the basic functionality of the system.
"""

import requests
import time
import os
from pathlib import Path

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_document_ingestion():
    """Test document ingestion"""
    print("Testing document ingestion...")
    pdf_path = "Solid Waste Management Rules 2026.pdf"

    if not os.path.exists(pdf_path):
        print(f"✗ PDF file not found: {pdf_path}")
        return False

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            response = requests.post("http://localhost:8000/ingest", files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Document ingested successfully: {result}")
            return True
        else:
            print(f"✗ Document ingestion failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Document ingestion error: {e}")
        return False

def test_query():
    """Test querying the system with different query types"""
    print("Testing query functionality...")

    test_cases = [
        {"query": "What is solid waste?", "type": "general"},
        {"query": "What are the rules for waste management?", "type": "factual"},
        {"query": "What are the penalties for improper disposal?", "type": "analytical"},
        {"query": "Summarize the key waste management regulations", "type": "summarization"}
    ]

    for test_case in test_cases:
        query = test_case["query"]
        query_type = test_case["type"]

        try:
            payload = {
                "query": query,
                "top_k": 3,
                "query_type": query_type
            }
            response = requests.post("http://localhost:8000/query", json=payload)

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Query '{query}' ({query_type}) successful")
                print(f"  Answer preview: {result['answer'][:100]}...")
                print(f"  Sources found: {len(result['sources'])}")
                print(f"  Confidence: {result.get('confidence', 'N/A'):.3f}")
                print(f"  Processing time: {result.get('processing_time', 'N/A'):.2f}s")
            else:
                print(f"✗ Query '{query}' ({query_type}) failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Query error: {e}")
            return False

        time.sleep(1)  # Brief pause between queries

    return True

def test_list_documents():
    """Test listing documents"""
    print("Testing document listing...")
    try:
        response = requests.get("http://localhost:8000/documents")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Found {len(result['documents'])} documents")
            return True
        else:
            print(f"✗ Document listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Document listing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Multimodal RAG System Tests")
    print("=" * 50)

    # Wait for server to be ready
    print("Waiting for server to start...")
    time.sleep(5)

    tests = [
        test_health_check,
        test_document_ingestion,
        test_list_documents,
        test_query
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())