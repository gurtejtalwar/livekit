import logging
from livekit.agents import cli, WorkerOptions

from app.services.agent.inbound_agent import inbound_entrypoint

logger = logging.getLogger("inbound-agent")

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(
        entrypoint_fnc=inbound_entrypoint,
        agent_name="inbound-agent",
        initialize_process_timeout=60  # This must match your dispatch rule
    ))
