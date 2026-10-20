class RecommendationsEngine:
    def generate_realtime_recommendations(self, session_status: dict):
        """
        Rule-based recommendations for immediate feedback.
        Input: Session status dict (people, emotions, attention, audio_context)
        Output: List of { type: 'alert' | 'warning' | 'info', priority: str, message: str, impact: str }
        """
        recs = []

        people = session_status.get("people", [])
        avg_att = session_status.get("avg_attention", 75.0)
        avg_eng = session_status.get("avg_engagement", 75.0)
        audio = session_status.get("audio_context", {})
        noise_db = audio.get("noise_db", -44.0)

        # 1. Low Attention Alert for individuals
        if people:
            for p in people:
                if p.get("attention", 100) < 40:
                    recs.append({
                        "type": "alert",
                        "priority": "High Priority",
                        "message": f"Student {p.get('id')} is distracted ({p.get('attention', 0)}%).",
                        "impact": "Prompt individual check-in"
                    })

        # 2. Global Attention / Disengagement Warning
        if len(people) >= 2 and avg_att < 60:
            recs.append({
                "type": "warning",
                "priority": "Medium Priority",
                "message": f"Class attention is low ({int(avg_att)}%). Launch a live poll or quiz to re-engage.",
                "impact": "+18% Engagement Recovery"
            })
        elif avg_eng >= 80:
            # High engagement context recommendation matching page 16 screenshot:
            # "Challenge students with harder problem - Impact: Deep Learning"
            recs.append({
                "type": "info",
                "priority": "High Priority",
                "message": "Challenge students with harder problem",
                "impact": "Deep Learning"
            })

        # 3. Audio / Classroom Management Cues
        if noise_db > -30:
            recs.append({
                "type": "warning",
                "priority": "Medium Priority",
                "message": f"Classroom ambient noise is high ({noise_db} dB). Pause lecture or request quiet.",
                "impact": "Acoustic Clarity"
            })
        elif audio.get("activity_type") == "silence" and avg_att < 50:
            recs.append({
                "type": "info",
                "priority": "Low Priority",
                "message": "Long silence observed with low attention. Propose a collaborative discussion.",
                "impact": "Active Discussion"
            })

        # Default positive status if all is well
        if not recs:
            recs.append({
                "type": "info",
                "priority": "Normal",
                "message": "Engagement is optimal. Keep going!",
                "impact": "Optimal Retention"
            })

        return recs
