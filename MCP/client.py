from chat_assistant import ChatAssistant
from mcp_client import MCPClient

client = MCPClient(["python", "mcp_server.py"])
client.start_server()

chat_assistant = ChatAssistant(client)
chat_assistant.run()