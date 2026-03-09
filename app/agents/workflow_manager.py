import logging
import asyncio
from typing import Dict, Type
from livekit.agents import AgentTask, llm
from app.agents.tools import TOOL_REGISTRY, ToolContext, load_knowledge_base

logger = logging.getLogger("workflow")

def _resolve_node_tools(agent_config, settings: dict) -> list:
    """Resolves tools specific to a node's settings."""
    resolved_tools = []
    
    # Tool Context
    kb = None
    if "knowledge_base_id" in settings:
        kb = load_knowledge_base(settings["knowledge_base_id"])
    elif agent_config.knowledge_base_id:
        # Fallback to agent's default KB if not overridden by node
        kb = load_knowledge_base(agent_config.knowledge_base_id)
        
    tool_ctx = ToolContext(agent_id=agent_config.agent_id, kb=kb)
    
    for tool_name in settings.get("tools", []):
        resolver = TOOL_REGISTRY.get(tool_name)
        if resolver:
            resolved_tools.append(resolver(tool_ctx))
        else:
            logger.warning(f"Unknown tool requested in workflow: {tool_name}")
            
    return resolved_tools

def _build_prompt(instructions: str, settings: dict, transitions: dict = None, global_nodes: list = None) -> str:
    """Builds the system prompt with few-shot examples and strict transition rules if present."""
    prompt = instructions + "\n\n"
    
    if transitions is not None or global_nodes:
        prompt += "### WORKFLOW ROUTING RULES\n"
        prompt += "You MUST route the conversation to the next step using the `transition` tool once your objective is complete. "
        prompt += "Here are the allowed transitions and their exact required reasons:\n"
        if transitions:
            for reason, next_node in transitions.items():
                prompt += f"- If {reason}, use next_state: '{next_node}' with reason: '{reason}'\n"
        if global_nodes:
            prompt += "You may also transition to any of these global nodes at any time if requested by the user:\n"
            for g_node in global_nodes:
                prompt += f"- next_state: '{g_node}'\n"
        prompt += "\n"
        
    examples = settings.get("examples", [])
    if examples:
        prompt += "### EXAMPLES\n"
        for ex in examples:
            prompt += f"User: {ex.get('user', '')}\nAssistant: {ex.get('assistant', '')}\n\n"
    return prompt

def _apply_voice_overrides(task: AgentTask, settings: dict):
    """Applies TTS overrides if defined in node settings."""
    overrides = settings.get("voice_overrides", {})
    if overrides and hasattr(task, 'agent') and task.agent and task.agent.tts:
         # Best-effort call to update options
         kwargs = {}
         if "speed" in overrides:
             kwargs["speed"] = overrides["speed"]
         if "emotion" in overrides:
             kwargs["emotion"] = [overrides["emotion"]]
         if kwargs:
             try:
                 task.agent.tts.update_options(**kwargs)
                 logger.debug(f"Applied voice overrides: {kwargs}")
             except Exception as e:
                 logger.error(f"Failed to apply voice overrides: {e}")
    
def create_rigid_task(state_name: str, state_config: dict, global_nodes: list, start_state: str) -> Type[AgentTask]:
    """Rigid mode: Agent strictly follows the graph edges."""
    
    class DynamicRigidTask(AgentTask):
        def __init__(self, agent, *args, **kwargs):
            # Resolve instructions with examples and exact transitions for the LLM
            settings = state_config.get("settings", {})
            transitions = state_config.get("transitions", {})
            full_instructions = _build_prompt(
                state_config.get("instructions", ""), 
                settings, 
                transitions, 
                global_nodes
            )
            kwargs["instructions"] = full_instructions
            if agent:
                if "llm" not in kwargs and hasattr(agent, "llm"): kwargs["llm"] = agent.llm
                if "tts" not in kwargs and hasattr(agent, "tts"): kwargs["tts"] = agent.tts
                if "stt" not in kwargs and hasattr(agent, "stt"): kwargs["stt"] = agent.stt
                
            super().__init__(*args, **kwargs)
            self.agent = agent
            # Pre-resolve tools for this node
            self._node_tools = _resolve_node_tools(agent.config, state_config.get("settings", {}))
            self.agent_config = agent.config
            
            # Important: Inject the node's tools onto this task instance
            # Depending on LiveKit SDK version, tools might need to be exposed via a method or property
            # For dynamic tools, usually we wrap them in a ToolSet or yield them in llm_node
            
        async def on_enter(self):
            logger.info(f"[Workflow] Entered rigid state: {state_name}")
            settings = state_config.get("settings", {})
            _apply_voice_overrides(self, settings)
            logger.debug(f"Rigid task '{state_name}' initialized with explicit routing prompt overhead.")
            
            # Fire an LLM generation for the first state so the agent greets the user natively
            if state_name == start_state:
                if hasattr(self.agent, "session"):
                    logger.info(f"[Workflow] Triggering initial LLM reply for start state: {state_name}")
                    self.agent.session.generate_reply()
            
        @llm.function_tool
        async def transition(self, next_state: str, reason: str):
            """
            Call this on each to move to the next state in the workflow.
            """
            allowed = list(state_config.get("transitions", {}).values()) + global_nodes
            if next_state not in allowed:
               return f"Error: Invalid transition. Must be one of {allowed}"
            
            logger.info(f"[Workflow] Transitioning to {next_state} because: {reason}")
            self.complete(next_state) 
            return f"Transitioning to {next_state}..."

        # Hack for dynamic tools on classes in LiveKit:
        # We need to expose the bound tools if the agent queries for them.
        def get_functions(self):
            funcs = super().get_functions() # Get the @llm.function_tool decorated methods (like transition)
            for t in self._node_tools:
                # Add our dynamic tools
                # Note: this assumes `t` is a callable that follows LiveKit's tool format
                if hasattr(t, '_llm_function'):
                    funcs.append(t)
            return funcs

    DynamicRigidTask.__name__ = f"{state_name}Task"
    return DynamicRigidTask

def create_flex_task(workflow_json: dict) -> Type[AgentTask]:
    """Flex mode: Entire graph is loaded into a single task with jumping tools."""
    
    class DynamicFlexTask(AgentTask):
        def __init__(self, agent, *args, **kwargs):
            self.current_node_name = workflow_json.get("start_state")
            # Resolve initial instructions
            node = workflow_json.get("states", {}).get(self.current_node_name, {})
            inst = node.get('instructions', '')
            settings = node.get('settings', {})
            
            base = f"You are currently at the '{self.current_node_name}' step. \nInstructions: {inst}\n"
            base += "You have tools available to jump to any other step in the flow if the user preemptively answers it."
            
            kwargs["instructions"] = _build_prompt(base, settings)
            if agent:
                if "llm" not in kwargs and hasattr(agent, "llm"): kwargs["llm"] = agent.llm
                if "tts" not in kwargs and hasattr(agent, "tts"): kwargs["tts"] = agent.tts
                if "stt" not in kwargs and hasattr(agent, "stt"): kwargs["stt"] = agent.stt
                
            super().__init__(*args, **kwargs)
            
            self.agent = agent
            self.agent_config = agent.config
            self._current_node_tools = []
        
        async def on_enter(self):
            logger.info(f"[Workflow] Started Flex workflow at: {self.current_node_name}")
            self._update_prompt_and_tools()
            
            # Fire an LLM generation to boot up the flex conversation
            if hasattr(self.agent, "session"):
                logger.info("[Workflow] Triggering initial LLM reply for flex workflow.")
                self.agent.session.generate_reply()

        def _update_prompt_and_tools(self):
            node = workflow_json["states"][self.current_node_name]
            settings = node.get("settings", {})
            
            # Apply tools and voice overrides
            self._current_node_tools = _resolve_node_tools(self.agent_config, settings)
            _apply_voice_overrides(self, settings)
            
            # Build prompt
            inst = node.get("instructions", "")
            base = f"You are currently at the '{self.current_node_name}' step. \nInstructions: {inst}\n"
            base += "You have tools available to jump to any other step in the flow if the user preemptively answers it."
            
            prompt = _build_prompt(base, settings)
            self._current_prompt = prompt

        @llm.function_tool
        async def jump_to_node(self, target_node: str, reason: str):
            """Use this tool to jump strictly to a different node in the conversation flow."""
            if target_node not in workflow_json.get("states", {}):
                return "Invalid node."
            
            logger.info(f"[Workflow] Flex Jump: {self.current_node_name} -> {target_node} ({reason})")
            self.current_node_name = target_node
            self._update_prompt_and_tools()
            
            # Since the context is immutable mid-flight, we return the new rules directly 
            # as the tool's result, injecting it into the active conversation path
            return f"Jump successful. You MUST now follow these new instructions strictly: {self._current_prompt}"

        def get_functions(self):
            funcs = super().get_functions()
            for t in self._current_node_tools:
                if hasattr(t, '_llm_function'):
                    funcs.append(t)
            return funcs

    return DynamicFlexTask

async def build_and_run_workflow(agent, workflow_json: dict):
    """Parses JSON, builds the graph, and runs the TaskGroup engine asynchronously."""
    mode = workflow_json.get("mode", "rigid")
    global_nodes = workflow_json.get("global_nodes", [])
    states = workflow_json.get("states", {})
    start_state = workflow_json.get("start_state")
    
    if not states or not start_state:
        logger.error("Workflow JSON is missing 'states' or 'start_state'.")
        return

    logger.info(f"[Workflow] Starting workflow engine in '{mode}' mode from '{start_state}'.")
    
    if mode == "rigid":
        task_classes = {}
        for state_name, config in states.items():
            task_classes[state_name] = create_rigid_task(state_name, config, global_nodes, start_state)

        current_state = start_state
        logger.info(f"[Workflow] Running Rigid Graph initialized with {len(task_classes)} states.")
        
        while current_state:
            if current_state not in task_classes:
                logger.warning(f"State '{current_state}' not found in workflow states, exiting.")
                break
                
            logger.info(f"[Workflow] Activating node: {current_state}")
            TaskCls = task_classes[current_state]
            
            # Ensure context persists across tasks without manual injection
            task = TaskCls(agent, chat_ctx=agent.chat_ctx.copy())
            
            try:
                # Suspends root agent, hands over session to this node
                # Execution yields back here with `next_state` when task completes via `self.complete(next_state)`
                current_state = await task 
            except Exception as e:
                logger.error(f"[Workflow] Error executing state '{current_state}': {e}")
                break
                
        logger.info(f"[Workflow] Finished Rigid Workflow. Final state: {current_state}")

        
    elif mode == "flex":
        FlexTaskCls = create_flex_task(workflow_json)
        
        logger.info("[Workflow] Running Flex Task directly.")
        task = FlexTaskCls(agent, chat_ctx=agent.chat_ctx.copy())
        
        try:
            results = await task
            logger.info(f"[Workflow] Finished Flex Workflow. Results: {results}")
        except Exception as e:
            logger.error(f"[Workflow] Flex Task execution error: {e}")
