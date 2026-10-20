import os
from dotenv import load_dotenv

load_dotenv()

class InsightGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Could not initialize Groq client: {e}")

    def generate_classroom_insight(self, session_data: dict) -> dict:
        """
        Generates real-time AI insights for instructors based on multimodal session data.
        """
        avg_att = session_data.get("avg_attention", 75.0)
        avg_eng = session_data.get("avg_engagement", 75.0)
        present = session_data.get("present_count", 0)
        audio = session_data.get("audio_context", {})

        prompt = (
            f"Analyze classroom dynamics: Average Attention: {avg_att}%, "
            f"Engagement Score: {avg_eng}%, Present Students: {present}, "
            f"Noise Level: {audio.get('noise_db', -44)} dB, Activity: {audio.get('activity_type', 'normal discussion')}. "
            f"Provide a 1-sentence diagnostic insight and 1-sentence pedagogical action."
        )

        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert educational AI assistant. Provide brief, professional, and actionable insights for teachers. Return format: 'Insight: ... | Recommendation: ...'"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model="mixtral-8x7b-32768",
                    temperature=0.5,
                    max_tokens=150,
                )
                response_text = chat_completion.choices[0].message.content
                parts = response_text.split("|")
                insight = parts[0].replace("Insight:", "").strip() if len(parts) > 0 else response_text.strip()
                recommendation = parts[1].replace("Recommendation:", "").strip() if len(parts) > 1 else "Maintain active student engagement."
                return {
                    "insight": insight,
                    "recommendation": recommendation,
                    "source": "groq-mixtral"
                }
            except Exception as e:
                print(f"Groq Insight fallback: {e}")

        # Deterministic rule-based fallback aligning with pedagogical best practices
        if avg_att >= 80:
            insight = "Class engagement is high with stable eye contact and attentiveness."
            recommendation = "Good momentum! Introduce higher-order challenge questions."
        elif avg_att >= 60:
            insight = "Class status is stable with moderate participation."
            recommendation = "Maintain current pace and invite questions from quieter participants."
        else:
            insight = "Noticeable dip in visual attention across multiple students."
            recommendation = "Break up lecture with a 60-second peer discussion or quick poll."

        return {
            "insight": insight,
            "recommendation": recommendation,
            "source": "rule-based-engine"
        }
