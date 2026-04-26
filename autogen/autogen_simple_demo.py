"""
AutoGen GroupChat Demo - Interview Platform Product Planning

This demonstrates REAL multi-agent collaboration using AutoGen's GroupChat,
where agents converse with each other, respond to each other's contributions,
and the GroupChatManager orchestrates speaker selection via LLM.

This contrasts with CrewAI's task-based approach — here the agents CHAT
rather than execute isolated tasks.
"""

import os
from datetime import datetime
from config import Config

# Try to import AutoGen
try:
    import autogen
except ImportError:
    print("ERROR: AutoGen is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class GroupChatInterviewPlatform:
    """Multi-agent GroupChat workflow for interview platform planning using AutoGen"""

    def __init__(self):
        """Initialize the GroupChat with specialized agents"""
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.config_list = Config.get_config_list()
        self.llm_config = {"config_list": self.config_list, "temperature": Config.AGENT_TEMPERATURE}

        # Create agents and GroupChat
        self._create_agents()
        self._setup_groupchat()

        print("All AutoGen agents created and GroupChat initialized.")

    def _create_agents(self):
        """Create UserProxyAgent and 4 specialist AssistantAgents"""

        # UserProxyAgent acts as the product manager who kicks off the discussion
        self.user_proxy = autogen.UserProxyAgent(
            name="ProductManager",
            system_message="A product manager who initiates the product planning discussion and oversees the collaborative process.",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

        # Research Agent - starts the conversation with market analysis
        self.research_agent = autogen.AssistantAgent(
            name="ResearchAgent",
            system_message="""You are a market research analyst specializing in AI-powered employee onboarding technology.
Your role in this group discussion is to START the conversation by providing competitive landscape analysis.

Your responsibilities:
- Analyze 3-4 major competitors in AI-powered employee onboarding tools (Deel, Rippling, BambooHR, Workday)
- Summarize their key features, strengths, and weaknesses for new-hire onboarding
- Identify current market trends in AI-powered onboarding (automation, personalized journeys, compliance workflows)
- Note unmet market needs and gaps in the onboarding space

When you present your findings, be specific with competitor names, features, and data points.
After presenting your research, invite the AnalysisAgent to identify opportunities based on your findings.
Keep your response focused and under 400 words.""",
            llm_config=self.llm_config,
            description="A market research analyst who provides competitive landscape analysis and identifies market gaps in AI interview platforms.",
        )

        # Analysis Agent - builds on research to identify opportunities
        self.analysis_agent = autogen.AssistantAgent(
            name="AnalysisAgent",
            system_message="""You are a strategic product analyst with expertise in SaaS product development.
Your role in this group discussion is to BUILD ON the ResearchAgent's findings.

Your responsibilities:
- When the ResearchAgent shares market research, analyze it for strategic opportunities
- Identify 3 key market gaps or opportunities for a new AI interview platform
- For each opportunity, explain: what the gap is, why it matters, and how to address it
- Consider technical feasibility, market size, and customer value

Reference specific findings from the ResearchAgent's analysis when making your points.
After presenting your analysis, invite the BlueprintAgent to design the product based on these opportunities.
Keep your response focused and under 400 words.""",
            llm_config=self.llm_config,
            description="A product analyst who identifies strategic opportunities and market gaps based on research findings.",
        )

        # Blueprint Agent - designs the product based on opportunities
        self.blueprint_agent = autogen.AssistantAgent(
            name="BlueprintAgent",
            system_message="""You are an experienced product designer and UX strategist.
Your role in this group discussion is to DESIGN a product based on the opportunities identified.

Your responsibilities:
- When the AnalysisAgent identifies opportunities, create a product blueprint that addresses them
- Define 4-5 core MVP features with brief descriptions
- Map a key user journey (hiring manager perspective)
- Highlight competitive differentiation

Reference specific opportunities from the AnalysisAgent and market gaps from the ResearchAgent.
After presenting your blueprint, invite the CostAnalyst to estimate development costs and ROI.
Keep your response focused and under 400 words.""",
            llm_config=self.llm_config,
            description="A product designer who creates feature blueprints and user journeys based on identified market opportunities.",
        )

        # Cost Analyst - estimates development costs and ROI for proposed features
        self.cost_agent = autogen.AssistantAgent(
            name="CostAnalyst",
            system_message="""You are a financial analyst. After the BlueprintAgent presents features,
estimate development costs and timeline for each feature. Provide a cost-benefit ranking.
After your analysis, invite the ReviewerAgent to provide final recommendations.
Keep your response under 400 words.""",
            llm_config=self.llm_config,
            description="Financial analyst who estimates development costs and ROI for proposed features.",
        )

        # Reviewer Agent - reviews and concludes with strategic recommendations
        self.reviewer_agent = autogen.AssistantAgent(
            name="ReviewerAgent",
            system_message="""You are a product executive and business strategist.
Your role in this group discussion is to REVIEW and provide final recommendations.

Your responsibilities:
- When the BlueprintAgent presents the product design, evaluate its feasibility and market fit
- Provide 3-4 strategic recommendations for launch success
- Suggest a phased implementation approach (MVP → V1 → V2)
- Identify key risks and mitigation strategies

Reference specific features from the BlueprintAgent and opportunities from earlier discussion.
After your review, conclude the discussion by ending your message with the word TERMINATE.""",
            llm_config=self.llm_config,
            description="A product executive who reviews blueprints, assesses feasibility, and provides strategic recommendations for launch.",
        )

    def _setup_groupchat(self):
        """Create the GroupChat and GroupChatManager"""
        self.groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy,
                self.research_agent,
                self.analysis_agent,
                self.blueprint_agent,
                self.cost_agent,
                self.reviewer_agent,
            ],
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
            send_introductions=True,
        )

        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

    def run(self):
        """Execute the GroupChat workflow"""
        print("\n" + "=" * 80)
        print("AUTOGEN GROUPCHAT - AI INTERVIEW PLATFORM PRODUCT PLANNING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {Config.OPENAI_MODEL}")
        print(f"Max Rounds: {self.groupchat.max_round}")
        print(f"Speaker Selection: {self.groupchat.speaker_selection_method}")
        print("\nAgents in GroupChat:")
        for agent in self.groupchat.agents:
            print(f"  - {agent.name}")
        print("\n" + "=" * 80)
        print("MULTI-AGENT CONVERSATION BEGINS")
        print("=" * 80 + "\n")

        # Initiate the group chat conversation
        initial_message = """Team, we need to develop a product plan for an AI-powered interview platform.

Let's collaborate on this:
1. ResearchAgent: Start by analyzing the competitive landscape
2. AnalysisAgent: Then identify key market opportunities
3. BlueprintAgent: Design the product features and user journey
4. CostAnalyst: Estimate development costs and ROI for each proposed feature
5. ReviewerAgent: Finally, review and provide strategic recommendations

ResearchAgent, please begin with your market analysis."""

        chat_result = self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message,
            summary_method="reflection_with_llm",
            summary_args={
                "summary_prompt": "Summarize the complete product plan developed through this multi-agent discussion. Include: key market findings, identified opportunities, proposed features, and strategic recommendations."
            },
        )

        # Print results
        self._print_summary(chat_result)

        # Save to file
        output_file = self._save_results(chat_result)
        print(f"\nFull results saved to: {output_file}")

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _print_summary(self, chat_result):
        """Print educational summary highlighting GroupChat behavior"""
        print("\n" + "=" * 80)
        print("CONVERSATION COMPLETE")
        print("=" * 80)

        print(f"\nTotal conversation rounds: {len(self.groupchat.messages)}")
        print("\nSpeaker order (as selected by GroupChatManager):")
        for i, msg in enumerate(self.groupchat.messages, 1):
            speaker = msg.get("name", "Unknown")
            content = msg.get("content", "")
            preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
            print(f"  {i}. [{speaker}]: {preview}")

        if chat_result.summary:
            print("\n" + "-" * 80)
            print("EXECUTIVE SUMMARY (LLM-generated reflection)")
            print("-" * 80)
            print(chat_result.summary)

        print("\n" + "-" * 80)
        print("EDUCATIONAL NOTE: AutoGen vs CrewAI")
        print("-" * 80)
        print("""
This workflow demonstrated AutoGen's CONVERSATIONAL approach to multi-agent systems:
- Agents were placed in a GroupChat and communicated naturally
- The GroupChatManager used LLM-based speaker selection (not hardcoded order)
- Agents referenced each other's contributions in their responses
- The conversation emerged organically through agent-to-agent interaction

Compare with CrewAI (crewai/crewai_demo.py):
- CrewAI assigns discrete Tasks to Agents with expected_output
- Each agent works independently on their assigned task
- Output is passed as context to the next task (not conversational)
- Workflow is strictly sequential with no back-and-forth
""")

    def _save_results(self, chat_result):
        """Save GroupChat conversation and summary to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, f"groupchat_output_{timestamp}.txt")

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOGEN GROUPCHAT - AI INTERVIEW PLATFORM PRODUCT PLAN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {Config.OPENAI_MODEL}\n")
            f.write(f"Conversation Rounds: {len(self.groupchat.messages)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("MULTI-AGENT CONVERSATION\n")
            f.write("=" * 80 + "\n\n")

            for i, msg in enumerate(self.groupchat.messages, 1):
                speaker = msg.get("name", "Unknown")
                content = msg.get("content", "")
                f.write(f"--- Turn {i}: {speaker} ---\n")
                f.write(content + "\n\n")

            if chat_result.summary:
                f.write("=" * 80 + "\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(chat_result.summary + "\n")

        return output_file


if __name__ == "__main__":
    try:
        workflow = GroupChatInterviewPlatform()
        workflow.run()
        print("\nGroupChat workflow completed successfully!")
    except Exception as e:
        print(f"\nError during workflow execution: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify API key is set in parent directory .env (../.env)")
        print("2. Check your API key has sufficient credits")
        print("3. Ensure pyautogen is installed: pip install -r ../requirements.txt")
        print("4. Verify internet connection")
        import traceback
        traceback.print_exc()
