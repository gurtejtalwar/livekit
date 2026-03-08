import logging
import asyncio
from typing import Dict, Type
from livekit.agents import AgentTask, llm
from livekit.agents.beta.workflows import TaskGroup
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

def _build_prompt(instructions: str, settings: dict) -> str:
    """Builds the system prompt with few-shot examples if present."""
    prompt = instructions + "\n\n"
    examples = settings.get("examples", [])
    if examples:
        prompt += "Examples of how to handle this state:\n"
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

def create_rigid_task(state_name: str, state_config: dict, global_nodes: list) -> Type[AgentTask]:
    """Rigid mode: Agent strictly follows the graph edges."""
    
    class DynamicRigidTask(AgentTask):
        def __init__(self, agent, *args, **kwargs):
            # Resolve instructions for the base AgentTask
            kwargs["instructions"] = state_config.get("instructions", "")
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
            
            prompt = _build_prompt(state_config.get("instructions", ""), settings)
            
            if hasattr(self.agent, 'chat_ctx') and self.agent.chat_ctx:
                sys_msg = next((m for m in self.agent.chat_ctx.messages if m.role == "system"), None)
                if sys_msg:
                    sys_msg.content = prompt
                else:
                    self.agent.chat_ctx.messages.append(llm.ChatMessage(role="system", content=prompt))
            else:
                 logger.warning("Agent ChatContext missing; could not set instructions.")
            
        @llm.function_tool
        async def transition(self, next_state: str, reason: str):
            """
            Call this to move to the next state in the workflow.
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
            base = f"You are currently at the '{self.current_node_name}' step. \nInstructions: {node.get('instructions', '')}\n"
            base += "You have tools available to jump to any other step in the flow if the user preemptively answers it."
            
            kwargs["instructions"] = base
            super().__init__(*args, **kwargs)
            
            self.agent = agent
            self.agent_config = agent.config
            self._current_node_tools = []
        
        async def on_enter(self):
            logger.info(f"[Workflow] Started Flex workflow at: {self.current_node_name}")
            self._update_prompt_and_tools()

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
            if hasattr(self.agent, 'chat_ctx') and self.agent.chat_ctx:
                sys_msg = next((m for m in self.agent.chat_ctx.messages if m.role == "system"), None)
                if sys_msg:
                    sys_msg.content = prompt
                else:
                    self.agent.chat_ctx.messages.append(llm.ChatMessage(role="system", content=prompt))

        @llm.function_tool
        async def jump_to_node(self, target_node: str, reason: str):
            """Use this tool to jump strictly to a different node in the conversation flow."""
            if target_node not in workflow_json.get("states", {}):
                return "Invalid node."
            
            logger.info(f"[Workflow] Flex Jump: {self.current_node_name} -> {target_node} ({reason})")
            self.current_node_name = target_node
            self._update_prompt_and_tools()
            
            if hasattr(self.agent, 'chat_ctx') and self.agent.chat_ctx:
                self.agent.chat_ctx.messages.append(
                    llm.ChatMessage(role="system", content=f"Successfully jumped state to {target_node}.")
                )
            return f"Jumped to {target_node}."

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
            task_classes[state_name] = create_rigid_task(state_name, config, global_nodes)
            
        group = TaskGroup(chat_ctx=agent.chat_ctx)
        
        # We add ALL generated rigid tasks into the TaskGroup.
        # The first task added will start execution.
        first_task = True
        for state_name, TaskCls in task_classes.items():
            def make_task_factory(cls=TaskCls, agt=agent):
                return lambda: cls(agt)
                
            group.add(
                make_task_factory(),
                id=state_name,
                description=states[state_name].get("instructions", f"Step: {state_name}")[:100]
            )
            
        # The workflow takes over the session and handles the re/entry of tasks
        logger.info(f"[Workflow] Running Rigid TaskGroup initialized with {len(task_classes)} states.")
        results = await group
        logger.info(f"[Workflow] Finished Rigid Workflow. Results: {results}")

        
    elif mode == "flex":
        FlexTaskCls = create_flex_task(workflow_json)
        group = TaskGroup(chat_ctx=agent.chat_ctx)
        
        group.add(
            lambda: FlexTaskCls(agent),
            id="flex_router",
            description="Flex router state holding all logic"
        )
        
        logger.info("[Workflow] Running Flex TaskGroup.")
        results = await group
        logger.info(f"[Workflow] Finished Flex Workflow. Results: {results}")
