[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tn7aj7Yc)
# 🤖 Multi-Agent Systems Lab: AutoGen vs. CrewAI

## Basics and Fundamentals (!!Read this first!!)
Before diving into the lab, please read through the [BASICS.md](BASICS.md) file to understand key concepts about multi-agent systems, architectures, and communication patterns. This foundational knowledge will help you grasp the implementations in this lab.

## 📚 Lab Overview

This lab introduces **multi-agent systems** - where multiple AI agents collaborate to solve complex problems. You'll work with two popular frameworks (**AutoGen** and **CrewAI**) to build, compare, and understand how intelligent agents can work together.

### What You'll Learn
- How to design agents with specific roles and responsibilities
- How agents communicate and collaborate
- Differences between conversational (AutoGen) vs. task-based (CrewAI) approaches
- When to use each framework for different problem types

---

## 🎯 What Are We Building?

### Part 1: AutoGen - Product Planning Workflow
**Scenario:** Build a product plan for an AI-powered interview platform

Four agents collaborate in a **GroupChat** (LLM selects who speaks next):
1. **ResearchAgent** - Analyzes market competitors
2. **AnalysisAgent** - Identifies key opportunities
3. **BlueprintAgent** - Creates product design
4. **ReviewerAgent** - Provides recommendations

**Communication Style:** Conversational GroupChat (agents converse freely, reference each other's contributions, LLM-based speaker selection)

### Part 2: CrewAI - Travel Planning Workflow
**Scenario:** Plan a 5-day trip to Iceland

Four agents form a "crew":
1. **FlightAgent** - Researches flights
2. **HotelAgent** - Finds accommodations
3. **ItineraryAgent** - Creates daily plans
4. **BudgetAgent** - Calculates costs

**Communication Style:** Task-based (each agent completes assigned tasks)

---

## 🛠️ Technologies Used

### **AutoGen** (Microsoft)
- Framework for building multi-agent systems with LLMs
- Agents converse in a **GroupChat** managed by a **GroupChatManager**
- LLM-based speaker selection — the model decides who speaks next
- Great for iterative, conversational problem solving

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Create specialized agents
researcher = AssistantAgent(name="Researcher", system_message="...", llm_config=llm_config)
analyst = AssistantAgent(name="Analyst", system_message="...", llm_config=llm_config)
user_proxy = UserProxyAgent(name="Admin", human_input_mode="NEVER", code_execution_config=False)

# Assemble into a group chat
groupchat = GroupChat(agents=[user_proxy, researcher, analyst], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start the multi-agent conversation
user_proxy.initiate_chat(manager, message="Your task here")
```

### **CrewAI** (Crew Framework)
- High-level framework for orchestrating agent "crews"
- Task-based execution - clear inputs and outputs
- Built-in tools and structured workflows
- Great for sequential, goal-oriented tasks

```python
from crewai import Agent, Task, Crew

agent = Agent(role="...", goal="...", backstory="...")
task = Task(description="...", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

### **OpenAI API**
- Powers the language models both frameworks use
- GPT-4 or GPT-4-Turbo for intelligent reasoning

---

## 📖 Lab Manual

### Prerequisites
- Python 3.8+
- OpenAI API key (get from https://platform.openai.com/api-keys)
- `pip` package manager

### Quick Setup

**1. Configure API Key:**
```bash
# Copy template
cp .env.example .env

# Add your OpenAI API key
# Edit .env and add:
OPENAI_API_KEY=sk-your-api-key-here
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Verify Configuration:**
```bash
python shared_config.py
```

### Running the Demos

**AutoGen Demo:**
```bash
python autogen/autogen_simple_demo.py
```

**CrewAI Demo:**
```bash
python crewai/crewai_demo.py
```

---

## 📁 Project Structure

```
multi-agent/
├── README.md                          ← You are here (complete lab guide)
├── requirements.txt                   ← Install ALL dependencies from here
├── .env.example                       ← Copy to .env (don't commit!)
├── .env                               ← Your configuration (add API key here)
├── shared_config.py                   ← Unified config for both frameworks
│
├── autogen/
│   ├── config.py                      ← AutoGen configuration (uses shared_config)
│   └── autogen_simple_demo.py         ← RUN THIS: GroupChat demo
│
└── crewai/
    └── crewai_demo.py                 ← RUN THIS: Travel planning demo
```

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Setup Environment
```bash
# Create .env file with your API key
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY
```

### Step 2: Install Packages
```bash
pip install -r requirements.txt
```

### Step 3: Test Configuration
```bash
python shared_config.py
# Should show: ✅ Configuration validation passed!
```

### Step 4: Run First Demo
```bash
# Try AutoGen
python autogen/autogen_simple_demo.py

# OR try CrewAI
python crewai/crewai_demo.py
```

---

## 🔍 Understanding Multi-Agent Systems

### Key Concepts

**Agent:** An AI entity with a specific role, goal, and reasoning ability
```python
Agent(
    role="Flight Specialist",
    goal="Find the best flights for the trip",
    backstory="You have booked thousands of flights..."
)
```

**Task:** Work to be completed by an agent
```python
Task(
    description="Research flights from NYC to Reykjavik",
    agent=flight_agent,
    expected_output="List of flight options with prices"
)
```

**Workflow:** How agents interact and pass information
- **GroupChat** (AutoGen): Agents converse in a shared chat room, LLM picks the next speaker each turn
- **Sequential** (CrewAI): Each agent completes an assigned task, output passes to next agent
- **Parallel:** Multiple agents work simultaneously (advanced)

### Why Use Multiple Agents?
- **Specialization:** Each agent is expert in one area
- **Modularity:** Easy to add/remove agents
- **Scalability:** Handle complex problems by breaking them down
- **Transparency:** Understand reasoning at each step

---

## 📊 Comparison: AutoGen vs. CrewAI

| Aspect | AutoGen | CrewAI |
|--------|---------|--------|
| **Communication** | Conversational GroupChat | Task-based |
| **Workflow** | Emergent, LLM-orchestrated | Structured, sequential |
| **Orchestration** | GroupChatManager selects speakers | Crew executes tasks in order |
| **Setup** | More code, more control | Less code, simpler |
| **Best For** | Iterative, collaborative problem-solving | Clear, goal-oriented workflows |
| **Agent Autonomy** | High (agents converse freely) | Lower (follows task structure) |
| **Output Structure** | Unstructured conversation | Structured task outputs |
| **Learning Curve** | Steeper | Gentler |

### Choose AutoGen If:
- ✓ Problem requires iteration, debate, and refinement
- ✓ Agents need to converse and build on each other's ideas
- ✓ Speaker order should emerge dynamically (LLM-selected)
- ✓ You need fine-grained control over agent interactions

### Choose CrewAI If:
- ✓ Workflow is well-defined and sequential
- ✓ Each agent has clear inputs/outputs
- ✓ You want faster setup with less code
- ✓ Tasks are independent and composable

---

## 🎓 Lab Exercises

### Exercise 1: Run and Understand
1. Run `autogen/autogen_simple_demo.py`
2. Read the output - understand how agents interact
3. Run `crewai/crewai_demo.py`
4. Compare the communication styles

### Exercise 2: Modify Agent Behavior

The goal is to observe how changing an agent's persona affects the group conversation (AutoGen) or task output (CrewAI).

**AutoGen:** Edit `autogen/autogen_simple_demo.py` — modify the `ResearchAgent`'s `system_message`:
```python
# Find this in _create_agents() and change the focus area:
self.research_agent = autogen.AssistantAgent(
    name="ResearchAgent",
    system_message="""You are a market research analyst specializing in...
    # ← Try changing the focus: instead of AI interview platforms,
    #   focus on "AI-powered employee onboarding tools"
    #   or change the competitors to research (Deel, Rippling, BambooHR)
    """,
    ...
)
```
Run the demo again — observe how downstream agents (AnalysisAgent, BlueprintAgent) adapt their responses to the new research context without any changes to their own prompts.

**CrewAI:** Edit `crewai/crewai_demo.py` — modify the `create_flight_agent()` function:
```python
return Agent(
    role="Flight Specialist",
    goal=f"...",
    backstory="You are an experienced flight specialist..."
    # ← Try adding constraints to the backstory:
    #   "You always prioritize direct flights over connections."
    #   "You focus on budget airlines and cost savings above all."
    # Then observe how the flight recommendations change.
)
```

**Questions to answer:**
- How does one agent's changed behavior ripple through to other agents?
- In AutoGen, did the GroupChatManager still select speakers in the same order?
- In CrewAI, did the budget agent's calculations reflect the flight agent's new priorities?

### Exercise 3: Add a Fifth Agent

Add a new specialist to each framework and observe how it changes the group dynamic.

**AutoGen** — Add a `CostAnalyst` agent to the GroupChat in `autogen/autogen_simple_demo.py`:

1. Create the agent in `_create_agents()`:
```python
self.cost_agent = autogen.AssistantAgent(
    name="CostAnalyst",
    system_message="""You are a financial analyst. After the BlueprintAgent presents features,
estimate development costs and timeline for each feature. Provide a cost-benefit ranking.
After your analysis, invite the ReviewerAgent to provide final recommendations.
Keep your response under 400 words.""",
    llm_config=self.llm_config,
    description="Financial analyst who estimates development costs and ROI for proposed features.",
)
```

2. Add it to the `agents` list in `_setup_groupchat()` (between BlueprintAgent and ReviewerAgent)

3. Increase `max_round` from 8 to 10

4. Run and observe: Does the GroupChatManager select the CostAnalyst at the right time? Does the ReviewerAgent incorporate cost data into its recommendations?

**CrewAI** — Add a `LocalExpert` agent and task in `crewai/crewai_demo.py`:

1. Create a new agent function that knows local customs, tips, and safety info
2. Create a corresponding `Task` with a specific `expected_output`
3. Add both to the `Crew` — place the task between itinerary and budget
4. Run and observe: Does the budget agent account for the local expert's tips?

### Exercise 4: Custom Problem

Rewrite one of the demos for a completely different domain. Pick one:
- **Conference planning** (speakers, venues, schedule, sponsorship)
- **Software architecture** (requirements, design, implementation plan, risk assessment)
- **Marketing campaign** (audience research, messaging, channels, budget)

Steps:
1. Keep the same framework structure (GroupChat for AutoGen, Crew for CrewAI)
2. Change agent roles, system messages, and the initial prompt
3. Run both frameworks on the same problem
4. Compare: Which framework produced a more useful result for your chosen domain? Why?

---

## 🔧 Troubleshooting

### "OPENAI_API_KEY is not configured"
```bash
# Make sure .env file exists and has your key
cat .env

# If missing, create it:
cp .env.example .env
# Then edit and add your key
nano .env
```

### "ModuleNotFoundError: No module named 'shared_config'"
```bash
# Run from project root, not from subdirectories
cd /Users/pranavhharish/Desktop/IS-492/multi-agent
python crewai/crewai_demo.py
```

### "Invalid API key" Error
```bash
# Check your key is valid at:
# https://platform.openai.com/account/api-keys

# Make sure it's in .env without quotes:
OPENAI_API_KEY=sk-proj-xxxxx  # ✓ Correct
# OPENAI_API_KEY="sk-proj-xxxxx"  # ✗ Wrong (don't use quotes)
```

### "Rate limit exceeded"
- Wait a few minutes and try again
- Check your API usage: https://platform.openai.com/account/usage

---

## 📚 Additional Resources

### Documentation
- **[AutoGen Docs](https://microsoft.github.io/autogen/)** - Official AutoGen documentation
- **[CrewAI Docs](https://docs.crewai.com/)** - Official CrewAI documentation
- **[OpenAI API](https://platform.openai.com/docs/)** - API reference and guides

### Learning
- **[LLM Agent Systems](https://lilianweng.github.io/posts/2023-06-23-agent/)** - Deep dive into agent theory
- **[ReAct Prompting](https://arxiv.org/abs/2210.03629)** - How agents think and act
- **[Multi-Agent Collaboration](https://arxiv.org/abs/2306.03589)** - Research on agent cooperation

### Community
- **AutoGen:** GitHub discussions at microsoft/autogen
- **CrewAI:** GitHub issues at joaomdmoura/crewai

