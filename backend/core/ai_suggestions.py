import os
from dotenv import load_dotenv

load_dotenv()

class AISuggestionEngine:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Could not initialize Groq client in AISuggestionEngine: {e}")

    def generate_suggestions(self, session_data: dict):
        """
        Generates structured teaching suggestions for instructors.
        """
        avg_att = session_data.get("avg_attention", 75.0)
        avg_eng = session_data.get("avg_engagement", 75.0)
        audio = session_data.get("audio_context", {})

        prompt = (
            f"Class metrics: Attention={avg_att}%, Engagement={avg_eng}%, "
            f"Noise={audio.get('noise_db', -44)}dB, Activity={audio.get('activity_type', 'normal')}. "
            f"Provide 2 concise pedagogical suggestions with priority and expected impact."
        )

        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert AI teaching assistant. "
                                "Return JSON or lines format: 'Priority: High/Medium | Suggestion: ... | Impact: ...'"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.7,
                    max_tokens=200
                )
                response = completion.choices[0].message.content
                return self._parse_response(response)
            except Exception as e:
                print(f"AI Suggestion Error: {e}")
                return self._fallback_suggestions(session_data)

        return self._fallback_suggestions(session_data)

    def _parse_response(self, text: str):
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        suggestions = []
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                prio = parts[0].replace("Priority:", "").strip()
                sugg = parts[1].replace("Suggestion:", "").strip()
                imp = parts[2].replace("Impact:", "").strip() if len(parts) > 2 else "Engagement improvement"
                suggestions.append({"priority": prio, "suggestion": sugg, "impact": imp})
        if not suggestions:
            suggestions.append({
                "priority": "High Priority",
                "suggestion": "Challenge students with harder problem",
                "impact": "Deep Learning"
            })
        return suggestions

    def _fallback_suggestions(self, session_data: dict):
        avg_att = session_data.get("avg_attention", 75.0)
        if avg_att > 80:
            return [
                {
                    "priority": "High Priority",
                    "suggestion": "Challenge students with harder problem",
                    "impact": "Deep Learning"
                },
                {
                    "priority": "Medium Priority",
                    "suggestion": "Assign breakout pairs for concept debate",
                    "impact": "+15% Retention"
                }
            ]
        elif avg_att > 60:
            return [
                {
                    "priority": "Medium Priority",
                    "suggestion": "Ask a direct conceptual check-in question",
                    "impact": "Keeps passive listeners active"
                },
                {
                    "priority": "Low Priority",
                    "suggestion": "Transition to practical coding or live example",
                    "impact": "+10% Attention span"
                }
            ]
        else:
            return [
                {
                    "priority": "High Priority",
                    "suggestion": "Pause lecture and run an interactive quiz or poll",
                    "impact": "+25% Attention Recovery"
                },
                {
                    "priority": "High Priority",
                    "suggestion": "Take a 60-second stretch or question break",
                    "impact": "Reduces cognitive fatigue"
                }
            ]
