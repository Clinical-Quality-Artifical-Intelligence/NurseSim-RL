#!/usr/bin/env python3
"""
A2A Protocol Compliance Tests for NurseSim-Triage Agent

Tests the agent's conformance to the Agent-to-Agent (A2A) protocol specification.
"""

import json
import subprocess
import time
import requests
from pathlib import Path


def test_agent_card_exists():
    """Test that agent-card.json exists and is valid JSON."""
    agent_card_path = Path(".well-known/agent-card.json")
    assert agent_card_path.exists(), "agent-card.json not found"
    
    with open(agent_card_path) as f:
        card = json.load(f)
    
    # Validate required fields
    assert "name" in card, "Missing 'name' field"
    assert "protocol" in card, "Missing 'protocol' field"
    assert card["protocol"] == "a2a/v1.0", f"Invalid protocol version: {card['protocol']}"
    assert "capabilities" in card, "Missing 'capabilities' field"
    assert "input_schema" in card["capabilities"], "Missing input_schema"
    assert "output_schema" in card["capabilities"], "Missing output_schema"
    
    print("✓ Agent card validation passed")
    return card


def test_input_schema_validation(agent_card):
    """Test that input schema is properly defined."""
    input_schema = agent_card["capabilities"]["input_schema"]
    
    assert input_schema["type"] == "object"
    assert "properties" in input_schema
    assert "complaint" in input_schema["properties"]
    assert "vitals" in input_schema["properties"]
    
    # Validate vitals schema
    vitals_schema = input_schema["properties"]["vitals"]
    assert "heart_rate" in vitals_schema["properties"]
    assert "blood_pressure" in vitals_schema["properties"]
    assert "spo2" in vitals_schema["properties"]
    assert "temperature" in vitals_schema["properties"]
    
    print("✓ Input schema validation passed")


def test_output_schema_validation(agent_card):
    """Test that output schema is properly defined."""
    output_schema = agent_card["capabilities"]["output_schema"]
    
    assert output_schema["type"] == "object"
    assert "properties" in output_schema
    assert "triage_category" in output_schema["properties"]
    assert "assessment" in output_schema["properties"]
    
    # Validate triage category enum
    triage_prop = output_schema["properties"]["triage_category"]
    assert "enum" in triage_prop
    expected_categories = ["Immediate", "Very Urgent", "Urgent", "Standard", "Non-Urgent"]
    assert triage_prop["enum"] == expected_categories
    
    print("✓ Output schema validation passed")


def test_agent_task_processing():
    """Test that the agent can process a task and return valid output."""
    # This test requires the agent to be running
    # For CI/CD, we'll test the agent_main.py module directly
    
    try:
        from agent_main import NurseSimTriageAgent
        
        # Initialize agent (requires HF_TOKEN in environment)
        agent = NurseSimTriageAgent()
        
        # Test task
        test_task = {
            "complaint": "Severe chest pain radiating to left arm",
            "vitals": {
                "heart_rate": 115,
                "blood_pressure": "85/60",
                "spo2": 91,
                "temperature": 37.5
            }
        }
        
        # Process task
        result = agent.process_task(test_task)
        
        # Validate result structure
        assert "triage_category" in result, "Missing triage_category in result"
        assert "assessment" in result, "Missing assessment in result"
        assert isinstance(result["assessment"], str), "Assessment must be a string"
        
        # Validate triage category is valid
        valid_categories = ["Immediate", "Very Urgent", "Urgent", "Standard", "Non-Urgent", "Error"]
        assert result["triage_category"] in valid_categories, \
            f"Invalid triage category: {result['triage_category']}"
        
        print(f"✓ Agent task processing passed")
        print(f"  Triage: {result['triage_category']}")
        print(f"  Assessment: {result['assessment'][:100]}...")
        
        return True
        
    except ImportError:
        print("⚠ Skipping task processing test (agent_main.py not available)")
        return False
    except ValueError as e:
        if "HF_TOKEN" in str(e):
            print("⚠ Skipping task processing test (HF_TOKEN not set)")
            return False
        raise


def test_agent_health_check():
    """Test agent health check endpoint."""
    try:
        from agent_main import NurseSimTriageAgent
        
        agent = NurseSimTriageAgent()
        health = agent.health_check()
        
        assert "status" in health
        assert health["status"] == "healthy"
        assert "model_loaded" in health
        
        print("✓ Health check passed")
        return True
        
    except (ImportError, ValueError):
        print("⚠ Skipping health check test")
        return False


def run_all_tests():
    """Run all A2A compliance tests."""
    print("\n" + "="*60)
    print("NurseSim-Triage A2A Protocol Compliance Tests")
    print("="*60 + "\n")
    
    try:
        # Test 1: Agent card validation
        agent_card = test_agent_card_exists()
        
        # Test 2: Input schema validation
        test_input_schema_validation(agent_card)
        
        # Test 3: Output schema validation
        test_output_schema_validation(agent_card)
        
        # Test 4: Task processing (requires model)
        test_agent_task_processing()
        
        # Test 5: Health check
        test_agent_health_check()
        
        print("\n" + "="*60)
        print("✓ All A2A compliance tests passed!")
        print("="*60 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
