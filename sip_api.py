"""
SIP API for creating inbound trunks and dispatch rules.
Provides endpoints to manage SIP trunks and dispatch rules with phone numbers and agent names.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from livekit import api

load_dotenv(dotenv_path=".env.dev", override=True)

app = FastAPI(title="LiveKit SIP API", version="1.0.0")


class SIPSetupRequest(BaseModel):
    """Request model for creating inbound trunk and dispatch rule"""
    phone_number: str
    agent_name: str
    trunk_name: str = None
    room_prefix: str = "call-"
    krisp_enabled: bool = True


class SIPSetupResponse(BaseModel):
    """Response model for SIP setup"""
    trunk_id: str
    trunk_name: str
    phone_number: str
    dispatch_rule_id: str
    agent_name: str
    message: str


class DeleteResponse(BaseModel):
    """Response model for delete operations"""
    message: str
    deleted_items: dict


@app.post("/sip/setup", response_model=SIPSetupResponse)
async def setup_sip_trunk_and_rule(request: SIPSetupRequest):
    """
    Create an inbound SIP trunk and dispatch rule.
    
    Args:
        request: SIPSetupRequest containing phone_number and agent_name
        
    Returns:
        SIPSetupResponse with created trunk and dispatch rule details
    """
    try:
        livekit_api = api.LiveKitAPI()
        
        # Generate trunk name if not provided
        trunk_name = request.trunk_name or f"trunk-{request.phone_number.replace('+', '').replace('-', '')}"
        
        # Step 1: Create inbound trunk
        trunk_info = api.SIPInboundTrunkInfo(
            name=trunk_name,
            numbers=[request.phone_number],
            krisp_enabled=request.krisp_enabled,
        )
        
        trunk_request = api.CreateSIPInboundTrunkRequest(trunk=trunk_info)
        trunk = await livekit_api.sip.create_inbound_trunk(trunk_request)
        trunk_id = trunk.sip_trunk_id
        
        print(f"Created inbound trunk: {trunk_id} for phone number: {request.phone_number}")
        
        # Step 2: Create dispatch rule
        dispatch_rule = api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                room_prefix=request.room_prefix,
            )
        )
        
        dispatch_rule_info = api.SIPDispatchRuleInfo(
            rule=dispatch_rule,
            name=f"dispatch-{request.agent_name}",
            trunk_ids=[trunk_id],
            room_config=api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(
                    agent_name=request.agent_name,
                    metadata=f"Agent dispatch for {request.phone_number}",
                )]
            )
        )
        
        dispatch_request = api.CreateSIPDispatchRuleRequest(
            dispatch_rule=dispatch_rule_info
        )
        
        dispatch = await livekit_api.sip.create_sip_dispatch_rule(dispatch_request)
        dispatch_id = dispatch.sip_dispatch_rule_id
        
        print(f"Created dispatch rule: {dispatch_id} for agent: {request.agent_name}")
        
        await livekit_api.aclose()
        
        return SIPSetupResponse(
            trunk_id=trunk_id,
            trunk_name=trunk_name,
            phone_number=request.phone_number,
            dispatch_rule_id=dispatch_id,
            agent_name=request.agent_name,
            message=f"Successfully created SIP trunk and dispatch rule for {request.phone_number}"
        )
        
    except Exception as e:
        await livekit_api.aclose()
        raise HTTPException(status_code=500, detail=f"Failed to setup SIP trunk and rule: {str(e)}")


@app.get("/sip/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "LiveKit SIP API"}


@app.delete("/sip/trunk/{trunk_id}", response_model=DeleteResponse)
async def delete_trunk(trunk_id: str):
    """
    Delete a SIP trunk and all its associated dispatch rules.
    
    Args:
        trunk_id: The ID of the trunk to delete
        
    Returns:
        DeleteResponse with details of deleted items
    """
    try:
        livekit_api = api.LiveKitAPI()
        deleted_items = {
            "trunk_id": trunk_id,
            "dispatch_rules": []
        }
        
        # Step 1: List all dispatch rules to find ones using this trunk
        dispatch_rules = await livekit_api.sip.list_sip_dispatch_rules()
        
        for rule in dispatch_rules.dispatch_rules:
            if trunk_id in rule.trunk_ids:
                # Delete dispatch rule
                await livekit_api.sip.delete_sip_dispatch_rule(rule.sip_dispatch_rule_id)
                deleted_items["dispatch_rules"].append({
                    "dispatch_rule_id": rule.sip_dispatch_rule_id,
                    "name": rule.name
                })
                print(f"Deleted dispatch rule: {rule.sip_dispatch_rule_id}")
        
        # Step 2: Delete the trunk
        await livekit_api.sip.delete_inbound_trunk(trunk_id)
        print(f"Deleted inbound trunk: {trunk_id}")
        
        await livekit_api.aclose()
        
        return DeleteResponse(
            message=f"Successfully deleted trunk {trunk_id} and {len(deleted_items['dispatch_rules'])} associated dispatch rule(s)",
            deleted_items=deleted_items
        )
        
    except Exception as e:
        await livekit_api.aclose()
        raise HTTPException(status_code=500, detail=f"Failed to delete trunk: {str(e)}")


@app.delete("/sip/dispatch-rule/{dispatch_rule_id}", response_model=DeleteResponse)
async def delete_dispatch_rule(dispatch_rule_id: str):
    """
    Delete a specific dispatch rule.
    
    Args:
        dispatch_rule_id: The ID of the dispatch rule to delete
        
    Returns:
        DeleteResponse with details of deleted item
    """
    try:
        livekit_api = api.LiveKitAPI()
        
        await livekit_api.sip.delete_sip_dispatch_rule(dispatch_rule_id)
        print(f"Deleted dispatch rule: {dispatch_rule_id}")
        
        await livekit_api.aclose()
        
        return DeleteResponse(
            message=f"Successfully deleted dispatch rule {dispatch_rule_id}",
            deleted_items={"dispatch_rule_id": dispatch_rule_id}
        )
        
    except Exception as e:
        await livekit_api.aclose()
        raise HTTPException(status_code=500, detail=f"Failed to delete dispatch rule: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
