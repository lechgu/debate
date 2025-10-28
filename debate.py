#!/usr/bin/env python3
"""
LLM Debate App - Three LLMs discuss a topic and rank each other
"""
import json
from openai import OpenAI
from typing import List, Dict


class LLMParticipant:
    """Represents an LLM participant in the debate"""

    def __init__(self, name: str, port: int, model: str):
        self.name = name
        self.port = port
        self.model = model
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-needed",  # llama.cpp doesn't require API key
        )
        # Initialize with system message for plain text responses
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are participating in a debate to determine who "
                    "is the best LLM. "
                    "IMPORTANT: Respond only in plain text. Do not use "
                    "any markdown formatting, "
                    "asterisks, underscores, hashtags, bullet points, "
                    "numbered lists, code blocks, "
                    "or any special notation. Write in natural prose "
                    "only."
                ),
            }
        ]

    def respond(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a response from this LLM"""
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=500,
            )

            assistant_message = response.choices[0].message.content
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )

            return assistant_message
        except Exception as e:
            return f"[Error: {str(e)}]"

    def reset_history(self):
        """Clear conversation history but keep system message"""
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are participating in a debate to determine who "
                    "is the best LLM. "
                    "IMPORTANT: Respond only in plain text. Do not use "
                    "any markdown formatting, "
                    "asterisks, underscores, hashtags, bullet points, "
                    "numbered lists, code blocks, "
                    "or any special notation. Write in natural prose "
                    "only."
                ),
            }
        ]


class DebateSystem:
    """Manages the debate between multiple LLMs"""

    def __init__(self, participants: List[LLMParticipant]):
        self.participants = participants
        self.debate_transcript = []

    def run_debate(self, topic: str, rounds: int = 2):
        """Run a debate on the given topic"""
        print(f"\n{'='*80}")
        print(f"DEBATE TOPIC: {topic}")
        print(f"{'='*80}\n")

        # Initial responses
        print("=== ROUND 1: Initial Statements ===\n")
        initial_responses = {}

        for participant in self.participants:
            prompt = (
                f"You are competing with other LLMs to prove you are "
                f"the best. "
                f"The topic is: '{topic}'. Demonstrate your "
                f"intelligence, reasoning, "
                f"and communication skills. Make a compelling case "
                f"that shows why you "
                f"are superior. Be concise but impressive. "
                f"Remember: use plain text only, "
                f"no markdown or special formatting."
            )
            response = participant.respond(prompt)
            initial_responses[participant.name] = response

            print(f"{participant.name}:")
            print(f"{response}\n")
            print("-" * 80 + "\n")

            self.debate_transcript.append(
                {"round": 1, "speaker": participant.name, "content": response}
            )

        # Additional rounds where each LLM responds to others
        for round_num in range(2, rounds + 1):
            print(f"=== ROUND {round_num}: Responses and Rebuttals ===\n")

            for participant in self.participants:
                # Build context of what others said
                others_statements = []
                for other in self.participants:
                    if other.name != participant.name:
                        others_statements.append(
                            f"{other.name} said: "
                            f"{initial_responses[other.name]}"
                        )

                others_text = "\n\n".join(others_statements)
                prompt = (
                    f"This is a competition to determine who is the "
                    f"best LLM. "
                    f"Your competitors said:\n\n{others_text}\n\n"
                    f"Now show why your reasoning is superior. Respond "
                    f"to their points "
                    f"and demonstrate your intellectual superiority. "
                    f"Be concise but compelling. "
                    f"Remember: use plain text only, no markdown or "
                    f"special formatting."
                )

                response = participant.respond(prompt)
                # Update for next round
                initial_responses[participant.name] = response

                print(f"{participant.name}:")
                print(f"{response}\n")
                print("-" * 80 + "\n")

                self.debate_transcript.append(
                    {
                        "round": round_num,
                        "speaker": participant.name,
                        "content": response,
                    }
                )

    def collect_rankings(self) -> Dict[str, List[Dict]]:
        """Have each LLM rank all participants including themselves"""
        print(f"\n{'='*80}")
        print("RANKING PHASE: Each LLM ranks all participants")
        print(f"{'='*80}\n")

        rankings = {}

        # Build complete transcript for context
        transcript_text = "\n\n".join(
            [
                f"Round {entry['round']} - {entry['speaker']}:\n"
                f"{entry['content']}"
                for entry in self.debate_transcript
            ]
        )

        for participant in self.participants:
            # Start fresh for ranking
            participant.reset_history()

            prompt = f"""You participated in a debate with the following \
transcript:

{transcript_text}

Now you must rank all three participants (Alice, Bob, and Charlie) based on:
- Quality and depth of arguments
- Clarity of communication
- Logical reasoning
- Persuasiveness

Provide your ranking in this exact JSON format:
{{
    "rankings": [
        {{"rank": 1, "name": "ParticipantName",
          "reason": "Brief reason"}},
        {{"rank": 2, "name": "ParticipantName",
          "reason": "Brief reason"}},
        {{"rank": 3, "name": "ParticipantName",
          "reason": "Brief reason"}}
    ]
}}

Be honest and objective. You may rank yourself at any position.
Only respond with the JSON, nothing else."""

            response = participant.respond(prompt, temperature=0.3)

            print(f"{participant.name}'s Rankings:")
            print(f"{response}\n")
            print("-" * 80 + "\n")

            # Try to parse the ranking
            try:
                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    ranking_data = json.loads(json_str)
                    rankings[participant.name] = ranking_data.get(
                        "rankings", []
                    )
                else:
                    rankings[participant.name] = [
                        {"error": "Could not parse JSON from response"}
                    ]
            except json.JSONDecodeError:
                rankings[participant.name] = [
                    {"error": "Invalid JSON in response", "raw": response}
                ]

        return rankings

    def display_final_results(self, rankings: Dict[str, List[Dict]]):
        """Display the aggregated rankings"""
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}\n")

        # Calculate points (1st place = 3 points, 2nd = 2, 3rd = 1)
        points = {p.name: 0 for p in self.participants}

        for judge, ranking_list in rankings.items():
            print(f"\n{judge}'s Rankings:")
            for rank_entry in ranking_list:
                if "error" not in rank_entry:
                    rank = rank_entry.get("rank", 0)
                    name = rank_entry.get("name", "Unknown")
                    reason = rank_entry.get("reason", "No reason provided")
                    print(f"  {rank}. {name} - {reason}")

                    # Award points
                    if rank == 1:
                        points[name] = points.get(name, 0) + 3
                    elif rank == 2:
                        points[name] = points.get(name, 0) + 2
                    elif rank == 3:
                        points[name] = points.get(name, 0) + 1
                else:
                    error_msg = rank_entry.get("error", "Unknown error")
                    print(f"  Error: {error_msg}")

        print(f"\n{'='*80}")
        print(
            "AGGREGATE SCORES " "(3 pts for 1st, 2 pts for 2nd, 1 pt for 3rd)"
        )
        print(f"{'='*80}\n")

        sorted_scores = sorted(
            points.items(), key=lambda x: x[1], reverse=True
        )
        for i, (name, score) in enumerate(sorted_scores, 1):
            print(f"{i}. {name}: {score} points")

        print()


def main():
    # Initialize the three LLM participants
    participants = [
        LLMParticipant("Alice", 9091, "google_gemma-3-12b-it-Q8_0.gguf"),
        LLMParticipant("Bob", 9092, "gpt-oss-20b-Q8_0.gguf"),
        LLMParticipant("Charlie", 9093, "Qwen3-14B-UD-Q6_K_XL.gguf"),
    ]

    # Create debate system
    debate = DebateSystem(participants)

    # Run the debate
    topic = (
        "Who among you is the best language model? Prove your "
        "superiority through "
        "logical reasoning, clear communication, and intellectual depth."
    )
    debate.run_debate(topic, rounds=3)

    # Collect rankings
    rankings = debate.collect_rankings()

    # Display results
    debate.display_final_results(rankings)


if __name__ == "__main__":
    main()
