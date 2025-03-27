import os

import discord
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)

COGS = ["cogs.chat"]


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

    # loading cogs
    for cog in COGS:
        bot.load_extension(cog)
        print(f"Loaded {cog}")

    await bot.sync_commands()


bot.run(os.getenv("DISCORD_BOT_TOKEN"))
