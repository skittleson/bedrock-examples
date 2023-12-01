from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage

chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
print(chat(messages))


