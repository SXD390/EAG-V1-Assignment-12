/agent/model_manager.py
```
import os
import json
import yaml
import requests
from pathlib import Path
from google import genai
from google.genai.errors import ServerError
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
MODELS_JSON = ROOT / "config" / "models.json"
PROFILE_YAML = ROOT / "config" / "profiles.yaml"

class ModelManager:
    def __init__(self):
        self.config = json.loads(MODELS_JSON.read_text())
        self.profile = yaml.safe_load(PROFILE_YAML.read_text())

        self.text_model_key = self.profile["llm"]["text_generation"]
        self.model_info = self.config["models"][self.text_model_key]
        self.model_type = self.model_info["type"]

        # ✅ Gemini initialization with new library
        if self.model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)

    async def generate_text(self, prompt: str) -> str:
        if self.model_type == "gemini":
            return await self._gemini_generate(prompt)

        elif self.model_type == "ollama":
            return await self._ollama_generate(prompt)

        raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    async def _gemini_generate(self, prompt: str) -> str:
        try:
            # ✅ CORRECT: Use truly async method
            response = await self.client.aio.models.generate_content(
                model=self.model_info["model"],
                contents=prompt
            )
            return response.text.strip()

        except ServerError as e:
            # ✅ FIXED: Raise the exception instead of returning it
            raise e
        except Exception as e:
            # ✅ Handle other potential errors
            raise RuntimeError(f"Gemini generation failed: {str(e)}")

    async def _ollama_generate(self, prompt: str) -> str:
        try:
            # ✅ Use aiohttp for truly async requests
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.model_info["url"]["generate"],
                    json={"model": self.model_info["model"], "prompt": prompt, "stream": False}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
```

agent/enhanced_agent_loop.py
```
# enhanced_agent_loop.py - Enhanced agent loop with multi-directional routing
"""
Enhanced Agent Loop with Multi-Directional Routing Support

Extends the existing agent loop to support:
- Multi-directional agent routing with return_to parameter
- BrowserAgent integration
- Enhanced error handling and recovery
- Agent scope and priority management
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import re
import json
import time

# Import existing modules
from perception.perception import Perception, build_perception_input
from decision.decision import Decision, build_decision_input
from summarization.summarizer import Summarizer
from agent.contextManager import ContextManager
from agent.agentSession import AgentSession
from memory.memory_search import MemorySearch
from action.execute_step import execute_step_with_mode
from utils.utils import log_step, log_error, save_final_plan, log_json_block

# Import browser-related modules
from agent.browserAgent import BrowserAgent, create_browser_agent
from agent.browser_tasks import Browser

class Route:
    SUMMARIZE = "summarize"
    DECISION = "decision"
    BROWSER = "browser"  # New route for browser operations
    DELEGATE = "delegate"  # New route for agent delegation

class StepType:
    ROOT = "ROOT"
    CODE = "CODE"
    BROWSER = "BROWSER"  # New step type for browser operations

class AgentType(Enum):
    PERCEPTION = "perception"
    DECISION = "decision"
    EXECUTION = "execution"
    BROWSER = "browser"  # New agent type
    SUMMARIZER = "summarizer"

class AgentPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class RoutingContext:
    """Manages routing between agents"""
    
    def __init__(self):
        self.current_agent = AgentType.PERCEPTION
        self.agent_stack = []
        self.routing_history = []
        self.return_paths = {}
        
    def push_agent(self, agent_type, return_to=None):
        """Push current agent to stack and switch to new agent"""
        # Convert string to AgentType if needed
        if isinstance(agent_type, str):
            try:
                agent_type = AgentType(agent_type.lower())
            except ValueError:
                agent_type = AgentType.PERCEPTION
                
        self.agent_stack.append({
            "agent": self.current_agent,
            "return_to": return_to,
            "timestamp": datetime.now()
        })
        self.current_agent = agent_type
        
    def pop_agent(self):
        """Pop agent from stack and return to it"""
        if self.agent_stack:
            previous = self.agent_stack.pop()
            self.current_agent = previous["agent"]
            return self.current_agent.value if hasattr(self.current_agent, 'value') else str(self.current_agent)
        return None
        
    def get_current_agent(self):
        """Get the current active agent"""
        if hasattr(self.current_agent, 'value'):
            return self.current_agent.value
        return str(self.current_agent)
        
    def get_return_path(self) -> Optional[AgentType]:
        """Get the return path for current agent"""
        if self.agent_stack:
            return self.agent_stack[-1].get("return_to")
        return None

class EnhancedAgentLoop:
    """Enhanced agent loop with multi-directional routing support"""
    
    def __init__(self, perception_prompt, decision_prompt, summarizer_prompt, browser_prompt, multi_mcp, strategy="exploratory"):
        # Initialize existing components
        self.perception = Perception(perception_prompt)
        self.decision = Decision(decision_prompt, multi_mcp)
        self.summarizer = Summarizer(summarizer_prompt)
        self.browser = Browser(browser_prompt)

        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.status: str = "in_progress"
        self.final_output = None  # Initialize final output
        
        # Initialize execution state
        self.code_variants = {}
        self.next_step_id = "ROOT"
        self.d_out = None
        self.p_out = {}  # Initialize as empty dict instead of None
        # Initialize routing and agents
        self.routing_context = RoutingContext()
        self.logger = logging.getLogger("EnhancedAgentLoop")
        self.browser_agent = None  # Will be initialized when needed
        self.query = ""
    
    async def run(self, query: str, initial_agent: AgentType = AgentType.PERCEPTION):
        """Enhanced run method with multi-agent routing support"""
        try:
            self._initialize_session(query)
            
            # Set initial agent
            self.routing_context.current_agent = initial_agent
            
            # Always start with perception unless explicitly overridden
            if initial_agent == AgentType.PERCEPTION:
                return await self._run_perception_workflow(query)
            else:
                return await self._run_custom_workflow(query, initial_agent)
                
        except Exception as e:
            self.logger.error(f"Enhanced agent loop failed: {str(e)}")
            return await self._handle_critical_failure(str(e))
    
    async def _run_perception_workflow(self, query: str):
        """Enhanced perception workflow with routing support"""
        await self._run_initial_perception()

        if self._should_early_exit():
            return await self._summarize()

        # Let perception decide the next route
        route = self.p_out.get("route")
        if route == Route.BROWSER:
            # Initialize browser agent when needed
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id, self.multi_mcp)
            return await self._run_browser_workflow(query)
        elif route == Route.DECISION:
            return await self._run_decision_loop()
        elif route == Route.SUMMARIZE:
            return await self._summarize()
        else:
            log_error("🚩 Invalid perception route. Exiting.")
            return "Invalid route from perception"
    
    async def _run_browser_workflow(self, query: str):
        """Run workflow starting with browser agent"""
        try:
            # self.logger.info("Starting browser-first workflow")
            
            # Initialize browser agent if needed
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id)
                
            # Get browser context
            
            # Parse query into tasks using LLM
            tasks = await self.browser.generate_tasks(query)
            
            results = []
            error = None
            for task in tasks:
                # Skip error-only tasks
                if "error" in task and len(task) == 1:
                    error = task["error"]
                    continue
                
                # Execute task with automatic routing
                task["return_to"] = "perception"
                result = await self.browser_agent.execute_task(task)
                results.append(result)
                
                if task.get("action") == "get_session_snapshot":
                    self.browser.session_snapshot = result['mcp_result'].content[0].text
                
                # Try fallback on failure
                if not result.get("success") and "fallback" in task:
                    self.logger.info(f"Task failed, trying fallback")
                    task["fallback"]["return_to"] = "perception"
                    fallback_result = await self.browser_agent.execute_task(task["fallback"])
                    results.append(fallback_result)
                    if fallback_result.get("success"):
                        continue
                
            self.browser.results.extend(results)
            
            # Return control based on results
            if error or not results:
                return await self._return_to_perception([{
                    "success": False,
                    "error": error or "No valid browser tasks",
                    "agent": "browser"
                }])
            elif all(r.get("success") for r in results):
                # Success - continue with decision phase
                return await self._return_to_perception(results)
            else:
                # Some tasks failed - return to perception for reanalysis
                return await self._return_to_perception(results)
            
        except Exception as e:
            return await self._handle_agent_error("browser", str(e))
    
    async def _run_custom_workflow(self, query: str, initial_agent: AgentType):
        """Run custom workflow starting with specified agent"""
        try:
            self.logger.info(f"Starting custom workflow with {initial_agent}")
            
            # Route to the specified initial agent
            context = {"query": query}
            result = await self._route_to_agent(initial_agent, context)
            
            # Handle both dict and string results
            if isinstance(result, dict):
                # Continue with standard flow based on results
                if result.get("success"):
                    # If successful, continue to decision or summarization
                    if self.p_out and self.p_out.get("route") == Route.DECISION:
                        return await self._run_decision_loop()
                    else:
                        return await self._summarize()
                else:
                    return await self._handle_agent_error(str(initial_agent), result.get("error", "Unknown error"))
            else:
                # Handle string results - assume success
                return result
                
        except Exception as e:
            return await self._handle_agent_error("custom_workflow", str(e))
    
    async def _route_to_agent(self, target_agent: AgentType, context: Dict[str, Any]):
        """Generic agent routing method"""
        try:
            self.routing_context.push_agent(target_agent, self.routing_context.current_agent)
            
            if target_agent == AgentType.BROWSER:
                return await self._execute_browser_tasks(context)
            elif target_agent == AgentType.PERCEPTION:
                return await self._execute_perception_tasks(context)
            elif target_agent == AgentType.DECISION:
                return await self._execute_decision_tasks(context)
            elif target_agent == AgentType.EXECUTION:
                return await self._execute_execution_tasks(context)
            else:
                raise ValueError(f"Unknown target agent: {target_agent}")
                
        except Exception as e:
            return await self._handle_agent_error(str(target_agent), str(e))
    
    async def _extract_browser_tasks(self, query: str, error: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract browser-specific tasks from query using LLM

        Args:
            query (str): The user's query
            error (str, optional): Previous error for retry scenarios

        Returns:
            List[Dict[str, Any]]: List of browser tasks to execute
        """
        try:
            # Initialize task generator if needed
            if not hasattr(self, 'browser'):
                self.browser = Browser("prompts/browser_prompt.txt")
            
            # Get current browser context
            context = {}
            if self.browser_agent:
                context = {
                    "session_id": self.session_id,
                    "current_url": getattr(self.browser_agent, "current_url", None),
                    "last_action": getattr(self.browser_agent, "last_action", None),
                }
            
            # Parse query into tasks using LLM
            tasks = await self.browser.generate_tasks(query, context, error)
            
            if not isinstance(tasks, list):
                tasks = [tasks]
            
            # Ensure all tasks are dictionaries
            typed_tasks: List[Dict[str, Any]] = []
            for task in tasks:
                if isinstance(task, dict):
                    typed_tasks.append(task)
                else:
                    typed_tasks.append({"error": str(task)})
            
            return typed_tasks
            
        except Exception as e:
            self.logger.error(f"Failed to parse browser tasks: {e}")
            # Fallback to basic element detection
            return [{
                "action": "get_elements",
                "parameters": {"strict_mode": True, "viewport_mode": "visible"}
            }]
    
    # async def _requires_browser_operations(self, query: str) -> bool:
    #     """Check if query requires browser operations by attempting to parse it"""
    #     try:
    #         if not hasattr(self, 'browser_parser'):
    #             from agent.browser_parser import BrowserParser
    #             self.browser_parser = BrowserParser("prompts/browser_prompt.txt")
                
    #         # Try to parse into browser tasks
    #         tasks = await self.browser_parser.parse_to_tasks(query)
            
    #         # If we got valid tasks (not just error), it's a browser operation
    #         return any(not ("error" in task and len(task) == 1) for task in tasks)
            
    #     except Exception as e:
    #         self.logger.warning(f"Failed to check browser operations: {e}")
    #         # Fallback to keyword matching
    #         browser_keywords = [
    #             "click", "navigate", "browser", "webpage", "website", "url",
    #             "button", "link", "form", "input", "extract", "screenshot"
    #         ]
    #         return any(keyword in query.lower() for keyword in browser_keywords)
    
    async def _continue_with_decision(self, browser_results: List[Dict[str, Any]]):
        """Continue with decision phase after successful browser operations"""
        try:
            # first make these JSON-safe
            browser_results = self._serialize_browser_results(browser_results)
            
            # Update context with browser results
            self.ctx.add_browser_results(browser_results)
            
            # Run decision with enhanced context
            return await self._run_decision_loop()
            
        except Exception as e:
            return await self._handle_agent_error("decision_continuation", str(e))
        
    def _serialize_browser_results(self, browser_results: List[Any]) -> List[Dict[str, Any]]:
        """Convert CallToolResult or other objects to JSON-serializable dicts."""
        serial_results: List[Dict[str, Any]] = []
        for r in browser_results:
            # if it’s already a dict, use it directly
            if isinstance(r, dict):
                d = r.copy()
            # else if it has a __dict__, shallow-copy that
            elif hasattr(r, "__dict__"):
                d = r.__dict__.copy()
            # otherwise stringify the whole object
            else:
                d = {"result": str(r)}

            # coerce any non-serializable field to string
            for k, v in list(d.items()):
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    d[k] = str(v)

            serial_results.append(d)
        return serial_results
    
    async def _return_to_perception(self, browser_results: List[Dict[str, Any]]):
        """Return to perception with browser results for reanalysis"""
        try:
            # first make these JSON-safe
            browser_results = self._serialize_browser_results(browser_results)
            # Update context with browser results (including failures)
            self.ctx.add_browser_results(browser_results)
            
            # Re-run perception with updated context
            p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="browser_feedback")
            self.p_out = await self.perception.run(p_input, session=self.session)
            
            # Process new perception output
            if self.p_out.get("route") == Route.DECISION:
                return await self._run_decision_loop()
            elif self.p_out.get("route") == Route.BROWSER:
            # Initialize browser agent when needed
                if not self.browser_agent:
                    self.browser_agent = await create_browser_agent(self.session_id, self.multi_mcp)
                return await self._run_browser_workflow(self.query)
            else:
                return await self._summarize()
                
        except Exception as e:
            return await self._handle_agent_error("perception_return", str(e))
    
    async def _execute_browser_tasks(self, context: Dict[str, Any]):
        """Execute browser-specific tasks"""
        if not self.browser_agent:
            self.browser_agent = await create_browser_agent(self.session_id)
        
        query = context.get("query", "")
        error = context.get("error")
        
        # Get tasks using LLM parser
        tasks = await self._extract_browser_tasks(query, error)
        
        results = []
        for task in tasks:
            # Skip error messages
            if "error" in task and len(task) == 1:
                self.logger.warning(f"Skipping error task: {task['error']}")
                continue
                
            result = await self.browser_agent.execute_task(task)
            results.append(result)
            
            # If task failed and has fallback, try the fallback
            if not result.get("success") and "fallback" in task:
                self.logger.info(f"Task failed, trying fallback: {task['fallback']}")
                fallback_result = await self.browser_agent.execute_task(task["fallback"])
                results.append(fallback_result)
        
        return {"success": True, "results": results, "agent": "browser"}
    
    async def _execute_perception_tasks(self, context: Dict[str, Any]):
        """Execute perception tasks"""
        p_input = build_perception_input(
            context.get("query", ""), 
            self.memory, 
            self.ctx, 
            snapshot_type="routed_perception"        )
        result = await self.perception.run(p_input, session=self.session)
        return {"success": True, "results": result, "agent": "perception"}
    
    async def _execute_decision_tasks(self, context: Dict[str, Any]):
        """Execute decision tasks"""
        # Ensure we have perception output for decision input
        p_out = getattr(self, 'p_out', {})
        if not p_out:
            # If no perception output available, create minimal one
            p_out = {"route": "decision", "result_requirement": context.get("query", "")}
            
        d_input = build_decision_input(
            self.ctx,                  # ctx (first parameter)
            context.get("query", ""),  # query (second parameter)
            p_out,                     # p_out (third parameter)
            self.strategy              # strategy (fourth parameter)
        )
        result = await self.decision.run(d_input, session=self.session)
        return {"success": True, "results": result, "agent": "decision"}
    
    async def _execute_execution_tasks(self, context: Dict[str, Any]):
        """Execute execution tasks"""
        # This would integrate with the existing execution system
        return {"success": True, "results": "Execution completed", "agent": "execution"}
    
    async def _summarize_results(self, results: List[Dict[str, Any]]):
        """Summarize results from multiple agents"""
        try:
            summary_input = {
                "query": self.query,
                "results": results,
                "session_context": self.ctx.get_context_snapshot()
            }
            
            summary = await self.summarizer.summarize(summary_input, self.ctx, self.p_out, session=self.session)
            return summary
            
        except Exception as e:
            return f"Summarization failed: {str(e)}"
    
    async def _handle_agent_error(self, agent_name: str, error: str):
        """Handle errors from specific agents"""
        self.logger.error(f"Agent {agent_name} failed: {error}")
        
        # Try to recover by routing to a different agent
        if agent_name == "browser" and self.routing_context.current_agent != AgentType.PERCEPTION:            
            return await self._route_to_agent(AgentType.PERCEPTION, {
                "query": self.query,
                "error_context": f"Browser agent failed: {error}"
            })
        
        return f"Agent {agent_name} failed: {error}"
    
    async def _handle_critical_failure(self, error: str):
        """Handle critical failures that can't be recovered"""
        self.logger.critical(f"Critical failure in enhanced agent loop: {error}")
        return f"Critical system failure: {error}"
    
    # Include all existing methods from the original AgentLoop
    def _initialize_session(self, query):
        """Initialize session (existing method)"""
        self.session_id = str(uuid.uuid4())
        self.query = query
        self.session = AgentSession(
            session_id=self.session_id,
            original_query=query
        )
        
        self.memory = MemorySearch().search_memory(query)
        # Remove load() call as MemorySearch doesn't have this method
        
        self.ctx = ContextManager(session_id=self.session_id, original_query=query)
        
        log_step(f"🚀 Starting Enhanced Agent Session: {self.session_id}")
        log_step(f"📝 Query: {query}")
        log_step(f"🎯 Strategy: {self.strategy}")
    
    async def _run_initial_perception(self):
        """Run initial perception (existing method)"""
        # Ensure context is properly initialized
        if not hasattr(self, 'ctx') or not hasattr(self.ctx, 'graph'):
            self.logger.error(f"Context not properly initialized. Type: {type(getattr(self, 'ctx', None))}")
            raise ValueError("ContextManager not properly initialized")
        
        p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="initial")
        self.p_out = await self.perception.run(p_input, session=self.session)
        self.ctx.attach_perception(StepType.ROOT, self.p_out)
        log_json_block("📌 Initial Perception", self.p_out)
    
    def _should_early_exit(self):
        """Check for early exit conditions (existing method)"""
        return (self.p_out.get("original_goal_achieved") or 
                self.p_out.get("route") == Route.SUMMARIZE)
    
    async def _summarize(self):
        """Generate summary (existing method)"""
        if hasattr(self, 'ctx') and self.ctx:
            summary_input = {
                "session_id": self.session_id,
                "query": self.query,
                "context": self.ctx.get_context_snapshot()
            }
            # import pdb; pdb.set_trace()
            return await self.summarizer.summarize(self.query, self.ctx, self.p_out, session=self.session)
        return "Summary unavailable - no context"
    
    async def _run_decision_loop(self):
        """Run decision loop with step execution (complete implementation)"""
        # Ensure we have perception output for decision input
        p_out = getattr(self, 'p_out', {})
        if not p_out:
            # If no perception output available, create minimal one
            p_out = {"route": "decision", "result_requirement": self.query}
        
        d_input = build_decision_input(
            self.ctx,       # ctx (first parameter)
            self.query,     # query (second parameter)  
            p_out,          # p_out (third parameter)
            self.strategy   # strategy (fourth parameter)
        )
        self.d_out = await self.decision.run(d_input, session=self.session)
        
        from utils.utils import log_json_block
        log_json_block("📌 Decision Output", self.d_out)
        
        if not self.d_out or not self.d_out.get("plan_graph"):
            self.status = "failed"
            return
            
        # Extract decision output components
        self.code_variants = self.d_out.get("code_variants", {})
        self.next_step_id = self.d_out.get("next_step_id", "ROOT")
        
        # Add steps from plan graph to context
        plan_graph = self.d_out.get("plan_graph", {})
        for node in plan_graph.get("nodes", []):
            from agent.agent_loop3 import StepType
            self.ctx.add_step(
                step_id=node["id"],
                description=node["description"],
                step_type=StepType.CODE,
                from_node=StepType.ROOT
            )
        # Execute the planned steps
        await self._execute_steps_loop()
    
    async def _execute_steps_loop(self):
        """Execute the planned steps in a loop"""
        from agent.agent_loop3 import StepExecutionTracker, StepType, Route
        from action.execute_step import execute_step_with_mode
        from perception.perception import build_perception_input
        from utils.utils import log_step, log_error, log_json_block
        
        tracker = StepExecutionTracker(max_steps=12, max_retries=5)
        AUTO_EXECUTION_MODE = "fallback"

        while tracker.should_continue():
            tracker.increment()
            log_step(f"🔁 Loop {tracker.tries} — Executing step {self.next_step_id}")

            if self.ctx.is_step_completed(self.next_step_id):
                log_step(f"✅ Step {self.next_step_id} already completed. Skipping.")
                self.next_step_id = self._pick_next_step(self.ctx)
                continue

            retry_step_id = tracker.retry_step_id(self.next_step_id)
            success = await execute_step_with_mode(
                retry_step_id,
                self.code_variants,
                self.ctx,
                AUTO_EXECUTION_MODE,
                self.session,
                self.multi_mcp
            )

            if not success:
                self.ctx.mark_step_failed(self.next_step_id, "All fallback variants failed")
                tracker.record_failure(self.next_step_id)

                if tracker.has_exceeded_retries(self.next_step_id):
                    if self.next_step_id == StepType.ROOT:
                        if tracker.register_root_failure():
                            log_error("🚨 ROOT failed too many times. Halting execution.")
                            return
                    else:
                        log_error(f"⚠️ Step {self.next_step_id} failed too many times. Forcing replan.")
                        self.next_step_id = StepType.ROOT
                continue

            self.ctx.mark_step_completed(self.next_step_id)

            # 🔍 Perception after execution
            p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="step_result")
            self.p_out = await self.perception.run(p_input, session=self.session)

            self.ctx.attach_perception(self.next_step_id, self.p_out)
            log_json_block(f"📌 Perception output ({self.next_step_id})", self.p_out)
            self.ctx._print_graph(depth=3)

            if self.p_out.get("original_goal_achieved") or self.p_out.get("route") == Route.SUMMARIZE:
                self.status = "success"
                self.final_output = await self._summarize()
                return

            if self.p_out.get("route") != Route.DECISION:
                log_error("🚩 Invalid route from perception. Exiting.")
                return

            # 🔁 Decision again
            d_input = build_decision_input(self.ctx, self.query, self.p_out, self.strategy)
            d_out = await self.decision.run(d_input, session=self.session)

            log_json_block(f"📌 Decision Output ({tracker.tries})", d_out)

            self.next_step_id = d_out["next_step_id"]
            self.code_variants = d_out["code_variants"]
            plan_graph = d_out["plan_graph"]
            self.update_plan_graph(self.ctx, plan_graph, self.next_step_id)
    
    def _pick_next_step(self, ctx) -> str:
        """Pick the next pending step from the context graph"""
        from agent.agent_loop3 import StepType
        for node_id in ctx.graph.nodes:
            node = ctx.graph.nodes[node_id]["data"]
            if node.status == "pending":
                return node.index
        return StepType.ROOT
    
    def update_plan_graph(self, ctx, plan_graph, from_step_id):
        """Update the plan graph with new nodes"""
        from agent.agent_loop3 import StepType
        for node in plan_graph["nodes"]:
            step_id = node["id"]
            if step_id in ctx.graph.nodes:
                existing = ctx.graph.nodes[step_id]["data"]
                if existing.status != "pending":
                    continue
            ctx.add_step(step_id, description=node["description"], step_type=StepType.CODE, from_node=from_step_id)
    
    async def _handle_failure(self):
        """Handle failure (existing method)"""
        return "Task execution failed"
    
    async def _retry_with_timeout(self, task: Dict[str, Any], timeout_ms: int = 5000) -> Dict[str, Any]:
        """Retry a task with timeout and exponential backoff"""
        try:
            start_time = time.time()
            backoff = 100  # Start with 100ms
            max_backoff = 1000  # Cap at 1 second
            
            # Ensure browser agent is initialized
            if not self.browser_agent:
                self.browser_agent = await create_browser_agent(self.session_id)
            
            while (time.time() - start_time) * 1000 < timeout_ms:
                result = await self.browser_agent.execute_task(task)
                if result.get("success"):
                    return result
                
                # Exponential backoff
                await asyncio.sleep(backoff / 1000)  # Convert to seconds
                backoff = min(backoff * 2, max_backoff)
            
            return {
                "success": False,
                "error": f"Task timed out after {timeout_ms}ms",
                "agent": "browser"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Retry failed: {str(e)}",
                "agent": "browser"
            }
            
    # async def _handle_browser_error(self, error: str, task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """Handle browser agent errors with automatic recovery"""
    #     self.logger.error(f"Browser error: {error}")
        
    #     if not task:
    #         return {
    #             "success": False,
    #             "error": error,
    #             "agent": "browser"
    #         }
            
    #     try:
    #         # Ensure browser agent is initialized
    #         if not self.browser_agent:
    #             self.browser_agent = await create_browser_agent(self.session_id)
            
    #         # Extract key info
    #         action = task.get("action", "").lower()
    #         parameters = task.get("parameters", {})
            
    #         # Handle specific error cases
    #         if "element not found" in error.lower() or "no such element" in error.lower():
    #             # Try getting elements again with extended viewport
    #             get_elements_task = {
    #                 "action": "get_elements",
    #                 "parameters": {"strict_mode": False, "viewport_mode": "all"}
    #             }
    #             return await self.browser_agent.execute_task(get_elements_task)
                
    #         elif "timeout" in error.lower():
    #             # Retry with extended timeout
    #             return await self._retry_with_timeout(task, timeout_ms=10000)
                
    #         elif "navigation" in error.lower():
    #             # Try navigation with fallback options
    #             if action in ["open_tab", "navigate"]:
    #                 url = parameters.get("url", "")
    #                 if url:
    #                     # Try alternate URL formations
    #                     if not url.startswith(("http://", "https://")):
    #                         alt_task = {**task, "parameters": {"url": f"https://{url}"}}
    #                         return await self.browser_agent.execute_task(alt_task)
                
    #         # Default error response
    #         return {
    #             "success": False,
    #             "error": error,
    #             "agent": "browser",
    #             "recovery_attempted": True
    #         }
            
    #     except Exception as e:
    #         return {
    #             "success": False,
    #             "error": f"Error recovery failed: {str(e)}",
    #             "agent": "browser"
    #         }
```

agent/contextManager.py
```

import time
from utils.utils import log_step, log_error, render_graph
import networkx as nx
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
from collections import defaultdict


@dataclass
class StepNode:
    index: str  # Now string to support labels like "0A", "0B"
    description: str
    type: str  # CODE, CONCLUDE, NOP
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    conclusion: Optional[str] = None
    error: Optional[str] = None
    perception: Optional[Dict[str, Any]] = None
    from_step: Optional[str] = None  # for debugging lineage


class ContextManager:
    def __init__(self, session_id: str, original_query: str):
        self.session_id = session_id
        self.original_query = original_query
        self.globals: Dict[str, Any] = {}
        self.session_memory: list[dict] = []  # Full memory, not compressed
        self.failed_nodes: list[str] = []     # Node labels of failed steps
        self.graph = nx.DiGraph()
        self.latest_node_id: Optional[str] = None
        self.executed_variants: Dict[str, Set[str]] = defaultdict(set)


        root_node = StepNode(index="ROOT", description=original_query, type="ROOT", status="completed")
        self.graph.add_node("ROOT", data=root_node)

    def add_step(self, step_id: str, description: str, step_type: str, from_node: Optional[str] = None, edge_type: str = "normal") -> str:
        step_node = StepNode(index=step_id, description=description, type=step_type, from_step=from_node)
        self.graph.add_node(step_id, data=step_node)
        if from_node:
            self.graph.add_edge(from_node, step_id, type=edge_type)
        self.latest_node_id = step_id
        # self._print_graph(depth=1)
        return step_id

    def is_step_completed(self, step_id: str) -> bool:
        node = self.graph.nodes.get(step_id, {}).get("data")
        return node is not None and node.status == "completed"


    def update_step_result(self, step_id: str, result: dict):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.result = result
        node.status = "completed"
        self._update_globals(result)
        # self._print_graph(depth=2)

    def mark_step_completed(self, step_id: str):
        if step_id in self.graph:
            node: StepNode = self.graph.nodes[step_id]["data"]
            node.status = "completed"


    def mark_step_failed(self, step_id: str, error_msg: str):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.status = "failed"
        node.error = error_msg
        self.failed_nodes.append(step_id)
        self.session_memory.append({
            "query": node.description,
            "result_requirement": "Tool failed",
            "solution_summary": str(error_msg)[:300]
        })
        # self._print_graph(depth=2)

    def attach_perception(self, step_id: str, perception: dict):
        if step_id not in self.graph.nodes:
            fallback_node = StepNode(index=step_id, description="Perception-only node", type="PERCEPTION")
            self.graph.add_node(step_id, data=fallback_node)
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.perception = perception
        if not perception.get("local_goal_achieved", True):
            self.failed_nodes.append(step_id)
        # self._print_graph(depth=2)

    def conclude(self, step_id: str, conclusion: str):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.status = "completed"
        node.conclusion = conclusion
        # self._print_graph(depth=2)

    def get_latest_node(self) -> Optional[str]:
        return self.latest_node_id

    def _update_globals(self, new_vars: Dict[str, Any]):
        for k, v in new_vars.items():
            if k in self.globals:
                versioned_key = f"{k}__{self.latest_node_id}"
                self.globals[versioned_key] = v
            else:
                self.globals[k] = v

    def _print_graph(self, depth: int = 1, only_if: bool = True):
        if only_if:
            render_graph(self.graph, depth=depth)

    def get_context_snapshot(self):
        def serialize_node_data(data):
            if hasattr(data, '__dict__'):
                safe: dict = {}
                for k, v in data.__dict__.items():
                    try:
                        # if this fails, v is not JSON serializable
                        json.dumps(v)
                        safe[k] = v
                    except (TypeError, ValueError):
                        # fallback: stringify it
                        safe[k] = str(v)
                return safe
            return data

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph, link="links")

        for node in graph_data["nodes"]:
            if "data" in node:
                node["data"] = serialize_node_data(node["data"])

        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "globals": self.globals,
            "memory": self.session_memory,
            "graph": graph_data,
        }

    def rename_subtree_from(self, from_step_id: str, suffix: str):
        if from_step_id not in self.graph:
            return
        to_rename = [from_step_id] + list(nx.descendants(self.graph, from_step_id))
        mapping = {}
        for old_id in to_rename:
            new_id = f"{old_id}{suffix}"
            node = self.graph.nodes[old_id]["data"]
            node.index = new_id
            self.graph.add_node(new_id, data=node)
            mapping[old_id] = new_id

        # Reconnect edges
        for old_src, old_tgt, attr in list(self.graph.edges(data=True)):
            new_src = mapping.get(old_src, old_src)
            new_tgt = mapping.get(old_tgt, old_tgt)
            if new_src != old_src or new_tgt != old_tgt:
                self.graph.add_edge(new_src, new_tgt, **attr)

        # Remove old nodes
        for old_id in to_rename:
            self.graph.remove_node(old_id)

        # Update failed_nodes list
        self.failed_nodes = [mapping.get(x, x) for x in self.failed_nodes]

    def attach_summary(self, summary: dict):
        """Attach summarizer output to session memory."""
        self.session_memory.append({
            "original_query": self.original_query,
            "result_requirement": "Final summary",
            "summarizer_summary": summary.get("summarizer_summary", summary if isinstance(summary, str) else ""),
            "confidence": summary.get("confidence", 0.95),
            "original_goal_achieved": True,
            "route": "summarize"
        })
    
    def add_browser_results(self, browser_results: List[Dict[str, Any]]):
        """Attach browser operation results to the context graph and update globals."""
        for i, result in enumerate(browser_results):
            # generate a unique step id
            step_id = f"BROWSER_{int(time.time() * 1000)}_{i}"
            # add it as a completed BROWSER step
            self.add_step(
                step_id=step_id,
                description=result.get("message", "browser_result"),
                step_type="BROWSER",
                from_node=self.latest_node_id
            )
            # mark it completed and merge any returned vars into globals
            self.update_step_result(step_id, result)

```

/agent/browser_tasks.py
```
"""Browser task generation and execution utilities"""
from pathlib import Path
import json
import logging
import sys
from typing import Dict, Any, Iterable, List, Optional
from browserMCP.mcp_tools import get_tools

from agent.model_manager import ModelManager

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('browser_agent.log')
    ]
)

class Browser:
    """Generates browser tasks from natural language using LLM"""
    
    def __init__(self, prompt_path: str = "prompts/browser_prompt.txt"):
        """Initialize the task generator
        
        Args:
            prompt_path (str): Path to the browser prompt template
        """
        self.prompt_path = prompt_path
        self.model = ModelManager()
        # Create logger with the class name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        self.session_snapshot = None  # Placeholder for session snapshot
        self.results = []  # Placeholder for results
        
    def _serialize_results(self, browser_results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all values in the results dict are JSON-serializable."""
        def _safe_serialize(value: Any) -> Any:
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                if isinstance(value, dict):
                    return {k: _safe_serialize(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_safe_serialize(v) for v in value]
                return str(value)

        return {k: _safe_serialize(v) for k, v in browser_results.items()}

    def _add_fallback_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add fallback tasks for common failure scenarios"""
        enhanced_tasks = []
        
        for task in tasks:
            # Skip tasks that already have fallbacks
            if "fallback" in task:
                enhanced_tasks.append(task)
                continue
                
            action = task.get("action", "").lower()
            parameters = task.get("parameters", {})
            
            # Add fallbacks based on action type
            if action in ["click_element_by_index", "input_text"]:
                # For element interactions, fallback to finding elements
                task["fallback"] = {
                    "action": "get_elements",
                    "parameters": {
                        "strict_mode": True,
                        "viewport_mode": "visible"
                    }
                }
            elif action in ["open_tab", "navigate"]:
                # For navigation, fallback to browser history
                task["fallback"] = {
                    "action": "go_back",
                    "parameters": {}
                }
            elif action in ["scroll_down", "scroll_up"]:
                # For scrolling, fallback to smaller amount
                amount = parameters.get("amount", 500)
                task["fallback"] = {
                    "action": task["action"],
                    "parameters": {"amount": amount // 2}
                }
                
            enhanced_tasks.append(task)
            
        return enhanced_tasks

    async def generate_tasks(
        self,
        query: str,
        error: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate browser tasks from natural language query
        
        Args:
            query: The user's query
            context: Optional browser context (current URL, state etc)
            error: Optional previous error message
            
        Returns:
            List of browser task dictionaries
        """
        try:
            # Load prompt template
            prompt = Path(self.prompt_path).read_text(encoding="utf-8")
            self.logger.info(f"Using prompt template from {self.prompt_path}")
            
            # Build input context
            input_json = {
                "query": query,
                "history": self.results,
                "session_snapshot": self.session_snapshot,
                "error": error or ""
            }
            
            input_json = self._serialize_results(input_json)
            
            full_prompt = prompt.format(context = json.dumps(input_json, indent=2), tools = get_tools()).strip()
            
            # self.logger.info(f"Generated prompt: {full_prompt}")
            
            # Get LLM response
            response = await self.model.generate_text(prompt=full_prompt)
            
            # self.logger.info(f"LLM response: {response}")
            
            try:
                # Remove any leading/trailing whitespace and backticks
                response = response.strip().strip("`")
                
                # Remove "json" or "JSON" from the start of the response if present
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                    
                tasks = json.loads(response).get("actions", [])
                
                if not isinstance(tasks, list):
                    tasks = [tasks]
                    
                # Add fallback tasks    
                tasks = self._add_fallback_tasks(tasks)
                
                return tasks
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {e}")
                return [{
                    "action": "get_elements",
                    "parameters": {"strict_mode": True, "viewport_mode": "visible"}
                }]
                
        except Exception as e:
            self.logger.error(f"Task generation failed: {e}")
            return [{
                "action": "get_elements",
                "parameters": {"strict_mode": True, "viewport_mode": "visible"}
            }]
```

agent/browser_parser.py
```
# """
# Browser Parser - LLM-based parser for browser actions
# """
# from pathlib import Path
# import json
# from typing import List, Dict, Any, Optional
# from agent.model_manager import ModelManager

# class BrowserParser:
#     """Parser that uses LLM to convert natural language to browser actions"""
    
#     def __init__(self, prompt_path: str, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
#         """Initialize browser parser
        
#         Args:
#             prompt_path (str): Path to browser prompt template
#             api_key (Optional[str], optional): API key. Defaults to None.
#             model (str, optional): Model to use. Defaults to "gemini-2.0-flash".
#         """
#         self.prompt_path = prompt_path
#         self.model = ModelManager()

#     async def parse_to_tasks(self, query: str, error: str = "", context: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
#         """Parse natural language query into browser tasks
        
#         Args:
#             query (str): User's query
#             context (Dict[str, Any], optional): Browser context. Defaults to None.
#             error (str, optional): Previous error for retry scenarios. Defaults to None.
        
#         Returns:
#             List[Dict[str, Any]]: List of browser tasks to execute
#         """
#         prompt_template = Path(self.prompt_path).read_text(encoding="utf-8")
        
#         # Build input for LLM
#         input_json = {
#             "query": query,
#             "context": context or {},
#             "error": error
#         }

#         full_prompt = (
#             f"{prompt_template.strip()}\n\n"
#             "```json\n"
#             f"{json.dumps(input_json, indent=2)}\n"
#             "```"
#         )

#         try:
#             response = await self.model.generate_text(prompt=full_prompt)
            
#             # Parse response as JSON
#             tasks = json.loads(response)
#             if not isinstance(tasks, list):
#                 tasks = [tasks]
                
#             return tasks
            
#         except Exception as e:
#             return [{
#                 "error": f"Failed to parse browser tasks: {str(e)}",
#                 "action": "get_elements",
#                 "parameters": {"strict_mode": True, "viewport_mode": "visible"}
#             }]
```

agent/browserAgent.py
```
# browserAgent.py - Specialized browser automation agent
"""
BrowserAgent: Specialized agent for browser-specific operations with multi-directional routing support

This agent handles:
- Browser automation tasks with proper element access
- Multi-directional routing between agents
- Smart error handling for browser operations
- Integration with existing Perception/Decision/Execution modules
"""

# Module level imports
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

# Global logger setup  
logger = logging.getLogger("BrowserAgent")

# Convenience functions for easy integration
async def create_browser_agent(
    session_id: Optional[str] = None,
    multi_mcp: Any = None
) -> 'BrowserAgent':
    """Create and initialize a BrowserAgent instance"""
    return BrowserAgent(session_id=session_id, multi_mcp=multi_mcp)

async def execute_browser_task(
    task: Dict[str, Any],
    session_id: Optional[str] = None,
    return_to: Optional[str] = None,
    multi_mcp: Any = None
) -> Dict[str, Any]:
    """Convenience function to execute a single browser task"""
    agent = await create_browser_agent(session_id, multi_mcp)
    return await agent.execute_task(task, return_to)

# Module exports - must be defined after the classes and functions
__all__: List[str] = [
    "BrowserAgent",
    "create_browser_agent",
    "execute_browser_task"
]

class BrowserAgent:
    """Specialized browser agent with multi-directional routing capabilities"""
    
    def __init__(self, session_id: Optional[str] = None, multi_mcp: Any = None):
        """Initialize BrowserAgent with session management"""
        self.session_id = session_id or "browser_agent_session"
        self.multi_mcp = multi_mcp
        
        # Routing configuration
        self.current_agent = "browserAgent"
        self.return_to: Optional[str] = None
        self.routing_stack: List[str] = []
        
        # Browser-specific configurations
        self.browser_context = {
            "structured_output": True,  # Always use structured output
            "strict_mode": True,        # Default to strict filtering
            "viewport_mode": "visible"  # Default to visible elements
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"BrowserAgent.{self.session_id}")
        
        if not multi_mcp:
            self.logger.warning("BrowserAgent initialized without MCP executor - browser operations will fail!")

    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool with proper error handling"""
        try:
            if not self.multi_mcp:
                raise ValueError("MCP executor not initialized")
            
            # Execute the tool directly through MultiMCP
            result = await self.multi_mcp.call_tool(tool_name, parameters)
            return result
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            raise

    async def execute_task(self, task: Dict[str, Any], return_to: Optional[str] = None) -> Dict[str, Any]:
        """Execute a browser-specific task with routing support"""
        try:
            self.return_to = return_to
            
            # Extract task information
            action = task.get("action", "").lower()
            parameters = task.get("parameters", {})
            
            self.logger.info(f"Executing browser task: {action}")
            
            # Route task based on action type
            if action in ["get_elements", "get_interactive_elements"]:
                return await self._handle_get_elements_task(action, parameters)
            elif action in ["click", "click_element", "click_element_by_index", "click_by_text"]:
                return await self._handle_click_task(parameters)
            elif action in ["input", "input_text", "type_text"]:
                return await self._handle_input_task(parameters)
            elif action in ["navigate", "go_to_url", "navigate_to_url", "open_tab"]:
                return await self._handle_navigation_task(action, parameters)
            elif action in ["extract", "get_content", "get_page_structure", "extract_page_content"]:
                return await self._handle_extraction_task(action, parameters)
            elif action in ["route_to", "delegate_to"]:
                return await self.route_to_agent(parameters.get("agent"), task)
            else:
                return await self._handle_generic_task(action, parameters)
            
        except Exception as e:
            self.logger.error(f"Error executing browser task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.current_agent
            }
    
    async def _handle_get_elements_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle element retrieval tasks with proper field access"""
        try:
            fixed_parameters = {
                "structured_output": True,
                "strict_mode": parameters.get("strict_mode", True),
                "visible_only": parameters.get("visible_only", True),
                "viewport_mode": parameters.get("viewport_mode", "visible")
            }
            
            result = await self._execute_mcp_tool("get_interactive_elements", fixed_parameters)
            # print(result)
            return {
                "success": True,
                "elements": result.content[0].text,
                "message": f"Found interactive elements",
                "agent": self.current_agent,
                "structured_output": True,
                "mcp_result": result
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get elements: {str(e)}",
                "agent": self.current_agent
            }

    async def _handle_click_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle click tasks with proper element ID handling"""
        try:
            element_id = parameters.get("index")
            text_to_find = parameters.get("text")
            
            if element_id is not None:
                result = await self._execute_mcp_tool("click_element_by_index", {"index": element_id})
                return {
                    "success": True,
                    "message": f"Clicked element with ID: {element_id}",
                    "element_id": element_id,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            elif text_to_find:
                # First get elements to find by text
                elements_result = await self._handle_get_elements_task("get_elements", {})
                if not elements_result.get("success"):
                    return elements_result
                
                # Find element with matching text
                import json
                elements_result = json.loads(elements_result.get("elements", "[]"))
                data = elements_result.get("nav", []) + elements_result.get("forms", [])
                for element in data:
                    if text_to_find.lower() in element.get("desc", "").lower():
                        result = await self._execute_mcp_tool("click_element_by_index", {"index": element["id"]})
                        return {
                            "success": True,
                            "message": f"Clicked element with text '{text_to_find}' (ID: {element['id']})",
                            "element_id": element["id"],
                            "agent": self.current_agent,
                            "mcp_result": result
                        }
                
                return {
                    "success": False,
                    "error": f"Element with text '{text_to_find}' not found",
                    "agent": self.current_agent
                }
            else:
                return {
                    "success": False,
                    "error": "No element ID provided",
                    "agent": self.current_agent
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to click element: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_input_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multiple input/text entry tasks"""
        try:
            element_id = parameters.get("index")
            text = parameters.get("text")
            result = None
            if element_id is not None:
                result = await self._execute_mcp_tool("input_text", {"index": element_id, "text": text})
                if not result.isError:
                    return {
                        "success": True,
                        "message": f"Entered '{text}' text in field with index: {element_id}",
                        "element_id": element_id,
                        "agent": self.current_agent,
                        "mcp_result": result
                    }
            return {
                "success": False,
                "error": "No element ID provided for input or the tool execution failed",
                "agent": self.current_agent,
                "mcp_result": result
            }
            # elif parameters:
            #     # First get elements to find by text
            #     elements_result = await self._handle_get_elements_task("get_elements", {})
            #     if not elements_result.get("success"):
            #         return elements_result
                
            #     # Find element with matching text
            #     import json
            #     elements_result = json.loads(elements_result.get("elements", "[]"))
            #     for param, value in parameters.items():
            #         for element in elements_result.get("forms", []):
            #             if param.lower() in element.get("desc", "").lower():
            #                 result = await self._execute_mcp_tool("input_text", {"index": element["id"], "text": value})
            # return {
            #     "success": True,
            #     "message": f"Input parameters '{json.dumps(parameters)}' to webpage",
            #     "element_id": element_id,
            #     "text": json.dumps(parameters),
            #     "agent": self.current_agent,
            #     "mcp_result": result
            # }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to input text: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_navigation_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation tasks"""
        try:
            url = parameters.get("url")
            
            if not url:
                return {
                    "success": False,
                    "error": "No URL provided for navigation",
                    "agent": self.current_agent
                }

            # Only try to open a new tab if requested
            if parameters.get("new_tab", True):
                try:
                    await self._execute_mcp_tool("open_tab", {})
                except Exception as e:
                    self.logger.error(f"Failed to open new tab (continuing with current tab): {e}")
            
            # Navigate to URL
            result = await self._execute_mcp_tool("go_to_url", {"url": url})
            return {
                "success": True,
                "message": f"Navigated to: {url}",
                "url": url,
                "agent": self.current_agent,
                "mcp_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to navigate: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_extraction_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content extraction tasks"""
        try:
            format_type = parameters.get("format", "markdown")
            
            if format_type == "structure":
                result = await self._execute_mcp_tool("get_enhanced_page_structure", {})
            else:
                result = await self._execute_mcp_tool("get_comprehensive_markdown", {})
            
            self.logger.info(f"Extracted content in {format_type} format")
            self.logger.info(f"Extraction result: {result}")  # Log first 100 chars
            
            return {
                "success": True,
                "content": result.content[0].text,
                "format": format_type,
                "agent": self.current_agent,
                "mcp_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract content: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_generic_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic browser tasks"""
        try:
            result = await self._execute_mcp_tool(action, parameters)
            return {
                "success": True,
                "message": f"Executed browser action: {action}",
                "action": action,
                "parameters": parameters,
                "agent": self.current_agent,
                "mcp_result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute {action}: {str(e)}",
                "agent": self.current_agent
            }
    
    # Routing methods for multi-directional agent communication
    async def route_to_agent(self, target_agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to another agent and manage routing stack"""
        if not target_agent:
            return {
                "success": False,
                "error": "No target agent specified",
                "agent": self.current_agent
            }

        try:
            # Add current agent to routing stack
            self.routing_stack.append(self.current_agent)
            
            # For now, return a mock response since we can't import other agents
            if target_agent == "perception":
                result = {
                    "success": True,
                    "message": "Routed to perception agent",
                    "agent": "perception",
                    "routed_from": self.current_agent
                }
            elif target_agent == "decision":
                result = {
                    "success": True,
                    "message": "Routed to decision agent",
                    "agent": "decision",
                    "routed_from": self.current_agent
                }
            elif target_agent == "execution":
                result = {
                    "success": True,
                    "message": "Routed to execution agent",
                    "agent": "execution",
                    "routed_from": self.current_agent
                }
            else:
                result = {
                    "success": False,
                    "error": f"Unknown target agent: {target_agent}",
                    "agent": self.current_agent
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to route to {target_agent}: {str(e)}",
                "agent": self.current_agent
            }
    
    async def return_control(self) -> Optional[str]:
        """Return control to the previous agent in the routing stack"""
        if self.routing_stack:
            return self.routing_stack.pop()
        return self.return_to
    
    def get_routing_context(self) -> Dict[str, Any]:
        """Get current routing context for debugging/monitoring"""
        return {
            "current_agent": self.current_agent,
            "return_to": self.return_to,
            "routing_stack": list(self.routing_stack),  # Create a copy
            "session_id": self.session_id
        }
```

agent/agent_loop3.py
```
import uuid
from datetime import datetime

from perception.perception import Perception, build_perception_input
from decision.decision import Decision, build_decision_input
from summarization.summarizer import Summarizer
from agent.contextManager import ContextManager
from agent.agentSession import AgentSession
from memory.memory_search import MemorySearch
from action.execute_step import execute_step_with_mode
from utils.utils import log_step, log_error, save_final_plan, log_json_block

class Route:
    SUMMARIZE = "summarize"
    DECISION = "decision"

class StepType:
    ROOT = "ROOT"
    CODE = "CODE"


class AgentLoop:
    def __init__(self, perception_prompt, decision_prompt, summarizer_prompt, multi_mcp, strategy="exploratory"):
        self.perception = Perception(perception_prompt)
        self.decision = Decision(decision_prompt, multi_mcp)
        self.summarizer = Summarizer(summarizer_prompt)
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.status: str = "in_progress"

    async def run(self, query: str):
        self._initialize_session(query)
        await self._run_initial_perception()

        if self._should_early_exit():
            return await self._summarize()

        # ✅ Missing early exit guard added
        if self.p_out.get("route") != Route.DECISION:
            log_error("🚩 Invalid perception route. Exiting.")
            return "Summary generation failed."

        await self._run_decision_loop()

        if self.status == "success":
            return self.final_output

        return await self._handle_failure()

    def _initialize_session(self, query):
        self.session_id = str(uuid.uuid4())
        self.ctx = ContextManager(self.session_id, query)
        self.session = AgentSession(self.session_id, query)
        self.query = query
        self.memory = MemorySearch().search_memory(query)
        self.ctx.globals = {"memory": self.memory}

    async def _run_initial_perception(self):
        p_input = build_perception_input(self.query, self.memory, self.ctx)
        self.p_out = await self.perception.run(p_input, session=self.session)

        self.ctx.add_step(step_id=StepType.ROOT, description="initial query", step_type=StepType.ROOT)
        self.ctx.mark_step_completed(StepType.ROOT)
        self.ctx.attach_perception(StepType.ROOT, self.p_out)

        log_json_block('📌 Perception output (ROOT)', self.p_out)
        self.ctx._print_graph(depth=2)

    def _should_early_exit(self) -> bool:
        return (
            self.p_out.get("original_goal_achieved") or
            self.p_out.get("route") == Route.SUMMARIZE
        )

    async def _summarize(self):
        return await self.summarizer.summarize(self.query, self.ctx, self.p_out, self.session)

    async def _run_decision_loop(self):
        """Executes initial decision and begins step execution."""
        d_input = build_decision_input(self.ctx, self.query, self.p_out, self.strategy)
        d_out = await self.decision.run(d_input, session=self.session)

        log_json_block("📌 Decision Output", d_out)

        self.code_variants = d_out["code_variants"]
        self.next_step_id = d_out["next_step_id"]

        for node in d_out["plan_graph"]["nodes"]:
            self.ctx.add_step(
                step_id=node["id"],
                description=node["description"],
                step_type=StepType.CODE,
                from_node=StepType.ROOT
            )

        await self._execute_steps_loop()

    async def _execute_steps_loop(self):
        tracker = StepExecutionTracker(max_steps=12, max_retries=5)
        AUTO_EXECUTION_MODE = "fallback"

        while tracker.should_continue():
            tracker.increment()
            log_step(f"🔁 Loop {tracker.tries} — Executing step {self.next_step_id}")

            if self.ctx.is_step_completed(self.next_step_id):
                log_step(f"✅ Step {self.next_step_id} already completed. Skipping.")
                self.next_step_id = self._pick_next_step(self.ctx)
                continue

            retry_step_id = tracker.retry_step_id(self.next_step_id)
            success = await execute_step_with_mode(
                retry_step_id,
                self.code_variants,
                self.ctx,
                AUTO_EXECUTION_MODE,
                self.session,
                self.multi_mcp
            )

            if not success:
                self.ctx.mark_step_failed(self.next_step_id, "All fallback variants failed")
                tracker.record_failure(self.next_step_id)

                if tracker.has_exceeded_retries(self.next_step_id):
                    if self.next_step_id == StepType.ROOT:
                        if tracker.register_root_failure():
                            log_error("🚨 ROOT failed too many times. Halting execution.")
                            return
                    else:
                        log_error(f"⚠️ Step {self.next_step_id} failed too many times. Forcing replan.")
                        self.next_step_id = StepType.ROOT
                continue

            self.ctx.mark_step_completed(self.next_step_id)

            # 🔍 Perception after execution
            p_input = build_perception_input(self.query, self.memory, self.ctx, snapshot_type="step_result")
            self.p_out = await self.perception.run(p_input, session=self.session)

            self.ctx.attach_perception(self.next_step_id, self.p_out)
            log_json_block(f"📌 Perception output ({self.next_step_id})", self.p_out)
            self.ctx._print_graph(depth=3)

            if self.p_out.get("original_goal_achieved") or self.p_out.get("route") == Route.SUMMARIZE:
                self.status = "success"
                self.final_output = await self._summarize()
                return

            if self.p_out.get("route") != Route.DECISION:
                log_error("🚩 Invalid route from perception. Exiting.")
                return

            # 🔁 Decision again
            d_input = build_decision_input(self.ctx, self.query, self.p_out, self.strategy)
            d_out = await self.decision.run(d_input, session=self.session)

            log_json_block(f"📌 Decision Output ({tracker.tries})", d_out)

            self.next_step_id = d_out["next_step_id"]
            self.code_variants = d_out["code_variants"]
            plan_graph = d_out["plan_graph"]
            self.update_plan_graph(self.ctx, plan_graph, self.next_step_id)


    async def _handle_failure(self):
        log_error(f"❌ Max steps reached. Halting at {self.next_step_id}")
        self.ctx._print_graph(depth=3)

        self.session.status = "failed"
        self.session.completed_at = datetime.utcnow().isoformat()

        save_final_plan(self.session_id, {
            "context": self.ctx.get_context_snapshot(),
            "session": self.session.to_json(),
            "status": "failed",
            "final_step_id": self.ctx.get_latest_node(),
            "reason": "Agent halted after max iterations or step failures.",
            "timestamp": datetime.utcnow().isoformat(),
            "original_query": self.ctx.original_query
        })

        return "⚠️ Agent halted after max iterations."

    def update_plan_graph(self, ctx, plan_graph, from_step_id):
        for node in plan_graph["nodes"]:
            step_id = node["id"]
            if step_id in ctx.graph.nodes:
                existing = ctx.graph.nodes[step_id]["data"]
                if existing.status != "pending":
                    continue
            ctx.add_step(step_id, description=node["description"], step_type=StepType.CODE, from_node=from_step_id)

    def _pick_next_step(self, ctx) -> str:
        for node_id in ctx.graph.nodes:
            node = ctx.graph.nodes[node_id]["data"]
            if node.status == "pending":
                return node.index
        return StepType.ROOT

    def _get_retry_step_id(self, step_id, failed_step_attempts):
        attempts = failed_step_attempts.get(step_id, 0)
        return f"{step_id}F{attempts}" if attempts > 0 else step_id


class StepExecutionTracker:
    def __init__(self, max_steps=12, max_retries=3):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.attempts = {}
        self.tries = 0
        self.root_failures = 0

    def increment(self):
        self.tries += 1

    def record_failure(self, step_id):
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1

    def retry_step_id(self, step_id):
        attempts = self.attempts.get(step_id, 0)
        return f"{step_id}F{attempts}" if attempts > 0 else step_id

    def should_continue(self):
        return self.tries < self.max_steps

    def has_exceeded_retries(self, step_id):
        return self.attempts.get(step_id, 0) >= self.max_retries

    def register_root_failure(self):
        self.root_failures += 1
        return self.root_failures >= 2
```

agent/agentSession.py
```
from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional
import uuid
import time
import json
from datetime import datetime

# Used by all snapshots
def current_utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

@dataclass
class ToolCode:
    tool_name: str
    tool_arguments: dict[str, Any]

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments
        }


@dataclass
class PerceptionSnapshot:
    run_id: str
    snapshot_type: str
    entities: list[str]
    result_requirement: str
    original_goal_achieved: bool
    reasoning: str
    local_goal_achieved: bool
    local_reasoning: str
    last_tooluse_summary: str
    solution_summary: str
    confidence: str
    route: str
    timestamp: str
    return_to: str = ""


@dataclass
class DecisionSnapshot:
    run_id: str
    input: dict
    output: dict
    next_step_id: str
    plan_graph: dict
    code_variants: dict
    timestamp: str = current_utc_ts()
    return_to: str = ""


@dataclass
class ExecutionSnapshot:
    run_id: str
    step_id: str
    variant_used: str
    code: str                     # NEW: full code string
    status: str                   # "success" or "error"
    result: Optional[Any]        # whatever run_user_code returns
    error: Optional[str]
    execution_time: str          # original start_timestamp
    total_time: str              # precomputed in run_user_code
    return_to: str = ""



@dataclass
class SummarizerSnapshot:
    run_id: str
    input: dict
    summary_output: str
    success: bool
    error: Optional[str]
    timestamp: str = current_utc_ts()
    return_to: str = ""



@dataclass
class Step:
    index: int
    description: str
    type: Literal["CODE", "CONCLUDE", "NOOP"]
    code: Optional[ToolCode] = None
    conclusion: Optional[str] = None
    execution_result: Optional[str] = None
    error: Optional[str] = None
    perception: Optional[PerceptionSnapshot] = None
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    attempts: int = 0
    was_replanned: bool = False
    parent_index: Optional[int] = None
    generated_vars: list[str] = None

    def to_dict(self):
        return {
            "index": self.index,
            "description": self.description,
            "type": self.type,
            "code": self.code.to_dict() if self.code else None,
            "conclusion": self.conclusion,
            "execution_result": self.execution_result,
            "error": self.error,
            "perception": self.perception.__dict__ if self.perception else None,
            "status": self.status,
            "attempts": self.attempts,
            "was_replanned": self.was_replanned,
            "parent_index": self.parent_index,
            "generated_vars": self.generated_vars if self.generated_vars else [],
        }


class AgentSession:
    def __init__(self, session_id: str, original_query: str):
        self.session_id = session_id
        self.original_query = original_query
        # self.perception: Optional[PerceptionSnapshot] = None
        self.perception_snapshots: list[PerceptionSnapshot] = []
        self.decision_snapshots: list[DecisionSnapshot] = []
        self.execution_snapshots: list[ExecutionSnapshot] = []
        self.summarizer_snapshots: list[SummarizerSnapshot] = []
        self.final_summary = None
        self.status: str = "in_progress"
        self.completed_at: Optional[str] = None

        self.plan_versions: list[dict[str, Any]] = []
        self.state = {
            "original_goal_achieved": False,
            "final_answer": None,
            "confidence": 0.0,
            "reasoning_note": "",
            "solution_summary": ""
            
        }

    # def add_perception(self, snapshot: PerceptionSnapshot):
    #     self.perception = snapshot
    def add_perception_snapshot(self, snapshot: PerceptionSnapshot):
        self.perception_snapshots.append(snapshot)

    def add_decision_snapshot(self, snapshot: DecisionSnapshot):
        self.decision_snapshots.append(snapshot)

    def add_execution_snapshot(self, snapshot: ExecutionSnapshot):
        self.execution_snapshots.append(snapshot)

    def add_summarizer_snapshot(self, snapshot: SummarizerSnapshot):
        self.summarizer_snapshots.append(snapshot)


    def add_plan_version(self, plan_texts: list[str], steps: list[Step]):
        plan = {
            "plan_text": plan_texts,
            "steps": steps.copy()
        }
        self.plan_versions.append(plan)
        return steps[0] if steps else None  # ✅ fix: return first Step

    def get_next_step_index(self) -> int:
        return sum(len(v["steps"]) for v in self.plan_versions)


    def to_json(self) -> dict:
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "perception_snapshots": [asdict(p) for p in self.perception_snapshots],
            "decision_snapshots": [asdict(d) for d in self.decision_snapshots],
            "execution_snapshots": [asdict(e) for e in self.execution_snapshots],
            "summarizer_snapshots": [asdict(s) for s in self.summarizer_snapshots],
            "final_summary": self.final_summary,
            "status": self.status,
            "completed_at": self.completed_at
        }

    def get_snapshot_summary(self):
        return {
            "session_id": self.session_id,
            "query": self.original_query,
            "final_plan": self.plan_versions[-1]["plan_text"] if self.plan_versions else [],
            "final_steps": [
                    asdict(s)
                    for version in self.plan_versions
                    for s in version["steps"]
                    if s.status == "completed"
                ],
            "final_executed_plan": [
                s.description
                for version in self.plan_versions
                for s in version["steps"]
                if s.status == "completed"
            ],
            "final_answer": self.state["final_answer"],
            "reasoning_note": self.state["reasoning_note"],
            "confidence": self.state["confidence"]
        }

    def mark_complete(self, perception: PerceptionSnapshot, final_answer: Optional[str] = None, fallback_confidence: float = 0.95):
        self.state.update({
            "original_goal_achieved": perception.original_goal_achieved,
            "final_answer": final_answer or perception.solution_summary,
            "confidence": perception.confidence or fallback_confidence,
            "reasoning_note": perception.reasoning,
            "solution_summary": perception.solution_summary,
        })
        # set the dedicated final_summary used in log
        self.final_summary = final_answer or perception.solution_summary



    def simulate_live(self, delay: float = 1.2):
        print("\n=== LIVE AGENT SESSION TRACE ===")
        print(f"Session ID: {self.session_id}")
        print(f"Query: {self.original_query}")
        time.sleep(delay)

        if self.perception_snapshots:
            print("\n[Perception 0] Initial ERORLL:")
            print(f"  {asdict(self.perception_snapshots)}")
            time.sleep(delay)

        for i, version in enumerate(self.plan_versions):
            print(f"\n[Decision Plan Text: V{i+1}]:")
            for j, p in enumerate(version["plan_text"]):
                print(f"  Step {j}: {p}")
            time.sleep(delay)

            for step in version["steps"]:
                print(f"\n[Step {step.index}] {step.description}")
                time.sleep(delay / 1.5)

                print(f"  Type: {step.type}")
                if step.code:
                    print(f"  Tool → {step.code.tool_name} | Args → {step.code.tool_arguments}")
                if step.execution_result:
                    print(f"  Execution Result: {step.execution_result}")
                if step.conclusion:
                    print(f"  Conclusion: {step.conclusion}")
                if step.error:
                    print(f"  Error: {step.error}")
                if step.perception:
                    print("  Perception ERORLL:")
                    for k, v in asdict(step.perception).items():
                        print(f"    {k}: {v}")
                print(f"  Status: {step.status}")
                if step.was_replanned:
                    print(f"  (Replanned from Step {step.parent_index})")
                if step.attempts > 1:
                    print(f"  Attempts: {step.attempts}")
                time.sleep(delay)

        print("\n[Session Snapshot]:")
        print(json.dumps(self.get_snapshot_summary(), indent=2))
```

