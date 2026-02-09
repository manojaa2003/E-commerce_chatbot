from groq import Groq

groq_client = Groq()

MAX_CHAT_TURNS = 4
SUMMARY_TRIGGER = 8


def summarize_conversation(recent_msgs):
    prompt = f"""
Summarize the following conversation.
Keep only key facts and user intent.

Conversation:
{chr(10).join(recent_msgs)}
"""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def fallback_chain(query, summary, recent_msgs):
    chat_history = "\n".join(recent_msgs) if recent_msgs else "None"

    system_prompt = f"""
You are a polite e-commerce assistant.

Your role:
- Handle unclear or unsupported requests
- Ask for clarification when needed
- Gently redirect to shopping-related help
- Never invent facts
- Keep responses short and friendly
Conversation summary:
{summary}

Recent conversation:
{chat_history}
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    answer = completion.choices[0].message.content.strip()

    # ---- UPDATE MEMORY ----
    recent_msgs.append(f"User: {query}")
    recent_msgs.append(f"Assistant: {answer}")

    if len(recent_msgs) > SUMMARY_TRIGGER * 2:
        new_summary = summarize_conversation(recent_msgs)
        summary = summary + "\n" + new_summary
        recent_msgs = recent_msgs[-MAX_CHAT_TURNS * 2 :]

    return answer, summary, recent_msgs
