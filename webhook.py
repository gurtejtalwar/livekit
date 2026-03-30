from fastapi import FastAPI, Request, HTTPException


app = FastAPI(title="LiveKit Webhook Receiver", version="1.0.0")

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Endpoint to receive webhook events from LiveKit.
    
    Args:
        request: Incoming HTTP request containing the webhook payload.
        
    Returns:
        Acknowledgment message upon successful receipt of the webhook.
    """
    try:
        payload = await request.json()
        # Process the webhook payload as needed
        print("Received webhook payload:", payload)
        
        return {"message": "Webhook received successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing webhook: {str(e)}")