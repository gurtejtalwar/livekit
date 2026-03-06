import asyncio
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.workflow_manager import create_rigid_task, create_flex_task, build_and_run_workflow
from app.agents.__init__ import AgentConfig, STTConfig, LLMConfig, TTSConfig

TEST_WORKFLOW_JSON = {
  "start_state": "Greetings",
  "mode": "rigid", 
  "global_nodes": ["Transfer_Call"],
  "states": {
    "Greetings": {
      "instructions": "Say 'Hello, this is Anna... May I speak with {customer_name}?'",
      "transitions": { "verified": "Identity", "unavailable": "Transfer_Call" },
      "settings": {
         "voice_overrides": { "emotion": "positive", "speed": 1.1 }
      }
    },
    "Identity": {
      "instructions": "Could you provide your full name and DOB?",
      "transitions": { "success": "Account_Menu" },
      "settings": {
          "tools": ["book_appointment"],
          "examples": [
            {"user": "It's John.", "assistant": "Thanks John, and your Date of Birth?"}
          ]
      }
    },
    "Transfer_Call": {
       "instructions": "Transferring your call now.",
       "transitions": {},
       "settings": {}
    }
  }
}

class MockChatContext:
    def __init__(self):
        class MockMessage:
            def __init__(self):
                self.content = ""
        self.messages = [MockMessage()]
        
    def append(self, msg):
        self.messages.append(msg)

class MockAgent:
    def __init__(self, config):
        self.config = config
        self.chat_ctx = MockChatContext()
        self.tts = None

async def test_workflow_classes():
    print("Testing dynamic Task generation...")
    # 1. Create a dummy AgentConfig
    dummy_config = AgentConfig(
        user_id="test_user",
        agent_id="test_agent",
        agent_name="test_name",
        system_prompt="Base prompt",
        stt=STTConfig(model="test", provider="test"),
        llm=LLMConfig(model="test", provider="test", max_tokens=100),
        tts=TTSConfig(model="test", provider="test", voice_id="test", emotion="test", speed=1, volume=1),
    )
    
    # 2. Test Rigid Mode creation
    GreetingsTaskClass = create_rigid_task("Greetings", TEST_WORKFLOW_JSON["states"]["Greetings"], TEST_WORKFLOW_JSON["global_nodes"])
    IdentityTaskClass = create_rigid_task("Identity", TEST_WORKFLOW_JSON["states"]["Identity"], TEST_WORKFLOW_JSON["global_nodes"])
    
    print(f"Created classes: {GreetingsTaskClass.__name__}, {IdentityTaskClass.__name__}")
    
    # 3. Test instantiated behavior
    mock_agent = MockAgent(dummy_config)
    greetings_task = GreetingsTaskClass(dummy_config)
    greetings_task.agent = mock_agent # Mock assignment since it's normally done by TaskGroup
    
    await greetings_task.on_enter()
    print(f"Prompt after entering Greetings via Rigid Mode (note voice_overrides applied safely if TTS was present):\n{mock_agent.chat_ctx.messages[0].content}")
    
    # Test valid transition
    res = await greetings_task.transition("Identity", "customer verified")
    print(f"Transition result to Identity: {res}")
    
    # Test global node transition
    res2 = await greetings_task.transition("Transfer_Call", "customer wants human")
    print(f"Transition result to Transfer_Call (Global Node): {res2}")
    
    # Check Identity task for few-shot prompt
    identity_task = IdentityTaskClass(dummy_config)
    identity_task.agent = mock_agent
    await identity_task.on_enter()
    print(f"\nPrompt after entering Identity via Rigid Mode (note few-shot examples):\n{mock_agent.chat_ctx.messages[0].content}")
    
    
    # 4. Test Flex Mode creation
    print("\n-------------------------------\nTesting Flex Task Generation...")
    TEST_WORKFLOW_JSON["mode"] = "flex"
    FlexTaskClass = create_flex_task(TEST_WORKFLOW_JSON)
    flex_task = FlexTaskClass(dummy_config)
    flex_task.agent = mock_agent
    
    await flex_task.on_enter()
    print(f"Prompt after entering Flex Mode (Starts at Greetings):\n{mock_agent.chat_ctx.messages[0].content}")
    
    res = await flex_task.jump_to_node("Identity", "Because user said their DOB")
    print(f"Flex Jump result: {res}")
    print(f"Prompt after Flex Jump:\n{mock_agent.chat_ctx.messages[0].content}")


if __name__ == "__main__":
    asyncio.run(test_workflow_classes())
