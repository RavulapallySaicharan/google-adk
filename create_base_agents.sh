#!/bin/bash

# Update this URL to your actual agent registration endpoint
AGENT_API_URL="http://localhost:5555/agents/"

# Trade Capture Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Trade Capture Agent",
    "agentDescription": "Captures and validates incoming trade data for further processing.",
    "category": "Finance",
    "goals": "Ingest trade data and perform initial validation.",
    "agentInputs": ["trade_data"]
}'

echo "Registered Trade Capture Agent"

# Compliance Check Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Compliance Check Agent",
    "agentDescription": "Performs compliance checks on trade data to ensure regulatory adherence.",
    "category": "Finance",
    "goals": "Check trade data for compliance with internal and external regulations.",
    "agentInputs": ["trade_data"]
}'

echo "Registered Compliance Check Agent"

# Trade Settlement Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Trade Settlement Agent",
    "agentDescription": "Handles the settlement of trades, ensuring proper transfer of securities and cash.",
    "category": "Finance",
    "goals": "Settle trades by coordinating with counterparties and clearing systems.",
    "agentInputs": ["trade_data"]
}'

echo "Registered Trade Settlement Agent"

# Reporting Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Reporting Agent",
    "agentDescription": "Generates regulatory and client reports post-settlement.",
    "category": "Finance",
    "goals": "Produce and distribute settlement and compliance reports.",
    "agentInputs": ["settlement_data"]
}'

echo "Registered Reporting Agent"

# Asset Servicing Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Asset Servicing Agent",
    "agentDescription": "Processes corporate actions, dividends, and other asset servicing events.",
    "category": "Finance",
    "goals": "Handle all asset servicing events and update records accordingly.",
    "agentInputs": ["asset_events"]
}'

echo "Registered Asset Servicing Agent"

# NAV Calculation Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "NAV Calculation Agent",
    "agentDescription": "Calculates Net Asset Value (NAV) for funds based on latest market and asset data.",
    "category": "Finance",
    "goals": "Compute NAV using validated fund and market data.",
    "agentInputs": ["fund_data"]
}'

echo "Registered NAV Calculation Agent"

# Reconciliation Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Reconciliation Agent",
    "agentDescription": "Reconciles asset servicing and NAV calculation results for consistency.",
    "category": "Finance",
    "goals": "Compare and reconcile outputs from asset servicing and NAV calculation.",
    "agentInputs": ["servicing_data", "nav_data"]
}'

echo "Registered Reconciliation Agent"

# Fund Accounting Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "Fund Accounting Agent",
    "agentDescription": "Performs fund accounting based on reconciled asset and NAV data.",
    "category": "Finance",
    "goals": "Execute fund accounting tasks and prepare final figures for reporting.",
    "agentInputs": ["reconciled_data"]
}'

echo "Registered Fund Accounting Agent"

# External Pricing API Agent
curl -X POST "$AGENT_API_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "agentName": "External Pricing API Agent",
    "agentDescription": "Fetches real-time security prices from an external pricing API for NAV calculation.",
    "category": "Finance",
    "goals": "Call the external pricing API and return the latest prices for the given securities.",
    "agentInputs": ["security_ids"]
}'

echo "Registered External Pricing API Agent" 