from foundry_local_sdk import Configuration, FoundryLocalManager

FoundryLocalManager.initialize(Configuration(app_name="my-app"))
manager=FoundryLocalManager.instance

model =manager.catalog.get_model("qwen3-8b")
model.download()
model.load()

client=model.get_chat_client()

response=client.complete_chat([
    {"role": "user", "content": "Merhaba"}
])

print(response.choices[0].message.content)

model.unload()