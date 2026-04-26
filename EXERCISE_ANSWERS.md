# Multi-Agent Systems Lab — Exercise Answers

**Setup:** Python 3.11 venv, Groq API (`llama-3.3-70b-versatile`) instead of OpenAI.

All four demo runs (Exercise 2 AutoGen + CrewAI, Exercise 3 AutoGen + CrewAI) were executed against live Groq inference; the analysis below is based on the actual generated outputs.

---

## Exercise 1: Run and Compare Communication Styles

| Dimension | AutoGen GroupChat | CrewAI Crew |
|---|---|---|
| **How it starts** | A single `user_proxy.initiate_chat(manager)` message kicks off a free-form conversation | `crew.kickoff()` runs a predefined ordered list of tasks |
| **Inter-agent communication** | Shared chat history — every agent sees everything everyone has said | Output of the previous task is passed as context into the next task |
| **Who picks the next speaker** | `GroupChatManager` uses an LLM to choose (`"auto"`), or `round_robin` for a fixed cycle | Determined by the order of the task list at design time |
| **Output shape** | Free-form conversation (a `messages` array) | Structured `expected_output` strings per task |
| **Failure mode** | If the LLM selects the wrong speaker, the conversation derails | A single failing task blocks the whole pipeline |

---

## Exercise 2: Modify Agent Behavior — How Does It Ripple?

### Modifications

- **AutoGen** ([autogen/autogen_simple_demo.py:59-65](autogen/autogen_simple_demo.py:59)) — `ResearchAgent` was repointed from "AI-powered recruitment / interview platforms" to "AI-powered employee onboarding tools" (competitors: Deel, Rippling, BambooHR, Workday).
- **CrewAI** ([crewai/crewai_demo.py:245-253](crewai/crewai_demo.py:245)) — `flight_agent` backstory now hard-codes two constraints: "always prioritize direct flights over connections" and "strongly favor budget airlines (PLAY, Norse Atlantic, Spirit, Frontier)."

### Question 1: How does one agent's changed behavior ripple through to other agents?

**AutoGen — dramatic, narrative ripple:**
- Once `ResearchAgent` pivoted to onboarding, `AnalysisAgent` immediately produced three onboarding-specific market gaps (industry-specific onboarding, HR-system integration, onboarding analytics). Nobody mentioned interviewing.
- `BlueprintAgent` designed a product called **OnboardGenie** — five MVP features all about onboarding.
- `ReviewerAgent` followed the onboarding direction completely.
- **Notable side effect:** even though the `ProductManager`'s initial message still said *"AI-powered interview platform,"* the entire team followed the `ResearchAgent`'s pivot. The final LLM-generated executive summary even contradicted itself: *"OnboardGenie, an AI-powered interview platform"* — the model fused the two contexts because both ended up in the shared chat history.

**CrewAI — quantitative, line-item ripple:**
- `FlightAgent` now recommends PLAY Airlines $349 direct (budget direct flight) as the headline pick.
- The downstream `BudgetAgent` produced budget tier $349, mid-range $485, luxury $612 — but the cost-saving tips section now leads with *"consider budget options for flights (like PLAY Airlines)"*, clearly inheriting the flight agent's bias.
- In the Exercise 3 5-agent rerun, the luxury tier even skipped the more expensive Delta $612 flight in favor of Icelandair $485 — showing that the "direct-flight-only" constraint had bled into recommendations across all tiers, not just budget.

### Question 2: Did the GroupChatManager still select speakers in the same order?

**Yes — order was unchanged at 4 agents.** The sequence stayed `ProductManager → ResearchAgent → AnalysisAgent → BlueprintAgent → ReviewerAgent`. The reason is that each agent's `system_message` ends with an explicit "invite the XxxAgent next" instruction, so the LLM speaker-selector follows those handoff cues rather than reasoning about content. Topic shifted radically; speaker order did not.

### Question 3: Did the BudgetAgent's calculations reflect the FlightAgent's new priorities?

**Partially.** The budget tier explicitly used PLAY $349 direct, and cost-saving tips proactively recommended PLAY Airlines. However, the mid and luxury tiers still followed a "more expensive = higher tier" pattern, because the budget task's `expected_output` requires three tiers (budget / mid-range / luxury). The model treated "budget direct flight" as a *feature of the budget tier* rather than as an override applied across all tiers.

---

## Exercise 3: Add a Fifth Agent

### Modifications

- **AutoGen**: Added a [`CostAnalyst`](autogen/autogen_simple_demo.py:113-122) agent between `BlueprintAgent` and `ReviewerAgent`; raised `max_round` from 8 to 10. Also updated `BlueprintAgent`'s system message to invite `CostAnalyst` next, and added `CostAnalyst` to the initial team kickoff message.
- **CrewAI**: Added a [`LocalExpert`](crewai/crewai_demo.py:301-316) agent and corresponding [task](crewai/crewai_demo.py:399-417) between the itinerary and budget tasks. The budget task description was updated to explicitly require: *"You MUST apply the LocalExpert's quantified daily cost adjustments directly to your meal, activity, and miscellaneous estimates."*

### Important finding: AutoGen 5-agent + Groq llama-3.3-70b is unstable with `auto` selection

On the first 5-agent run with the default `speaker_selection_method="auto"`, the LLM speaker selector broke down:
- `ResearchAgent` was skipped entirely.
- `AnalysisAgent` was chosen as the next speaker, but its content was actually market research (the role of `ResearchAgent`).
- `BlueprintAgent` was chosen next, but its content was opportunity analysis (the role of `AnalysisAgent`).
- The conversation terminated after 3 turns, far short of `max_round=10`.

This is a real-world signal that AutoGen's LLM-based speaker selection is sensitive to model capability — Groq `llama-3.3-70b-versatile` is not as reliable as GPT-4 in this role once the agent count grows past four.

**Fix:** switching `speaker_selection_method` to `"round_robin"` ([autogen/autogen_simple_demo.py:153](autogen/autogen_simple_demo.py:153)) made all 6 turns complete in the correct order. The trade-off is losing dynamic, content-aware speaker selection in exchange for a reliable execution pipeline.

### Did the GroupChatManager select the CostAnalyst at the right time?

With `round_robin` it is guaranteed by construction. With `auto` (and this model), based on the failure above, it almost certainly would not.

### Did the ReviewerAgent incorporate cost data into its recommendations?

**Yes, explicitly.** In the strategy section, `ReviewerAgent` quoted the *"highest estimated ROI"* directly from `CostAnalyst`'s estimates and used those numbers to decide which features go into the MVP (AI-Driven Feedback Loop and Personalized Content Recommendations). It also reused `CostAnalyst`'s development-time estimates (6–8 weeks, 4–6 weeks, etc.) to construct the phased MVP → V1 → V2 rollout plan.

### Did the BudgetAgent account for the LocalExpert's tips?

**Yes, very explicitly.** The final budget report contained two new sections that did not exist before:
- **"Local Insider Tips for Cost Savings"** — the five tips that `LocalExpert` produced (supermarket bakeries, off-peak Blue Lagoon, public transit, happy hour, etc.)
- **"Savings from Following LocalExpert's Advice"** — quantified per-tier savings: budget ~$390/person, mid-range ~$495/person, luxury ~$691/person.

CrewAI's "previous task output is passed as context to the next task" mechanism, combined with the explicit instruction in the budget task description, made the propagation deterministic and auditable.

---

## Three Key Takeaways

### 1. AutoGen is emergent; CrewAI is contractual.
In AutoGen, changing one agent's prompt reshapes the entire conversation because every agent sees the shared history — propagation is *automatic but uncontrolled*, and downstream agents can drift in surprising ways (e.g., the executive summary calling an onboarding tool an "interview platform"). In CrewAI, propagation is *opt-in*: you must explicitly write "use the LocalExpert's adjustments" into the budget task description, otherwise the influence does not transfer.

### 2. LLM-based speaker selection is fragile on smaller / open models.
Groq's `llama-3.3-70b-versatile` could not reliably perform `auto` speaker selection once five specialist agents were in the room. For production use of AutoGen GroupChat with this many agents, you either need a stronger model (GPT-4-class) or you give up `auto` and use `round_robin` / a state machine.

### 3. The cost of changing one agent is asymmetric.
- **AutoGen:** Changing one agent implicitly changes everyone's context — propagation is free, but you have no control over how far it spreads or whether it stays coherent.
- **CrewAI:** Changing one agent does not automatically affect downstream agents. To make influence flow, you must update the downstream task's `description` or `expected_output`. More work, but the propagation path is explicit and reviewable.

When designing a multi-agent system, this maps to a real architectural choice: do you want emergent collaboration (AutoGen) or auditable pipelines (CrewAI)?

---

## Files Modified in This Lab

| File | What changed |
|---|---|
| [.env](.env) | Created with `GROQ_API_KEY` and `GROQ_MODEL=llama-3.3-70b-versatile` |
| [autogen/autogen_simple_demo.py](autogen/autogen_simple_demo.py) | `ResearchAgent` retargeted to onboarding tools; added `CostAnalyst`; `max_round` 8→10; speaker selection switched to `round_robin`; initial message updated to introduce `CostAnalyst` |
| [crewai/crewai_demo.py](crewai/crewai_demo.py) | `flight_agent` backstory adds direct-flight + budget-airline constraints; added `create_local_expert_agent` and `create_local_expert_task`; budget task now explicitly applies LocalExpert adjustments; crew `agents`/`tasks` lists extended to include the new agent |

## How to Reproduce

```bash
# 1. Activate the prepared venv (Python 3.11)
source venv/bin/activate

# 2. Verify Groq config is loaded
python shared_config.py

# 3. Run the AutoGen 5-agent GroupChat
python autogen/autogen_simple_demo.py

# 4. Run the CrewAI 5-agent travel-planning crew
python crewai/crewai_demo.py
```
