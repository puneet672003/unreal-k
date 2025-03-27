import discord
from discord.ext import commands
from semantic_text_splitter import TextSplitter

from utils.ai import AI

splitter = TextSplitter(2000)
GUILD_IDS = [1045348811125571644]


class Chat(commands.Cog):
    def __init__(self, bot):
        self.threads = {}
        self.bot = bot
        self.ai = AI()

    @discord.application_command(guild_ids=GUILD_IDS)
    async def start(self, ctx: discord.ApplicationContext):
        author_id = str(ctx.author.id)

        if self.threads.get(author_id) is None:
            message = await ctx.send(f"Hi {ctx.author.mention}")
            channel = await message.create_thread(
                name=f"Session w {ctx.author.name}", auto_archive_duration=60
            )

            self.threads[author_id] = channel.id
            await ctx.respond("Done! Check new Thread.")
        else:
            await ctx.respond(f"You already have an ongoing session.")

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return

        author_id = str(message.author.id)
        thread_id = self.threads.get(author_id)

        if thread_id and message.channel.id == thread_id:
            async with message.channel.typing():
                try:
                    async for response in self.ai.stream_response(
                        message.content, thread_id
                    ):
                        content = response.content
                        tokens_used = response.usage_metadata["total_tokens"]

                        if len(content) > 2000:
                            chunks = splitter.chunks(content)
                            for chunk in chunks:
                                if chunk:
                                    await message.channel.send(chunk)
                        else:
                            await message.channel.send(content)

                        if self.ai.TOKEN_LIMIT - tokens_used < 1000:
                            embed = discord.Embed(
                                title="Memory limit reached",
                                description="> To accommodate new responses, older messages will be gradually removed.",
                                color=discord.Color.red(),
                            )
                            await message.channel.send(embed=embed)
                except Exception as e:
                    print(e)

                    embed = discord.Embed(
                        title="Error",
                        description="An Unknown error occured. Please create new session for new conversations!",
                        color=discord.Color.red(),
                    )
                    await message.channel.send(embed=embed)
                    del self.threads[author_id]


def setup(bot):
    bot.add_cog(Chat(bot))
