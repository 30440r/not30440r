from io import BytesIO
import traceback

import discord
from cogs.commands.misc.confidence import run_on_file
from data.services.guild_service import guild_service
from discord.commands import slash_command, Option
from discord.ext import commands
from utils.config import cfg
from utils.context import BlooContext, PromptData
from utils.logger import logger
from utils.permissions.checks import PermissionsFailure
from utils.permissions.permissions import permissions

from utils.context import BlooContext
from utils.config import cfg

class NeuralNet(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @slash_command(guild_ids=[cfg.guild_id], description="Neural Net Guess")
    async def neural_net(self, ctx: BlooContext):
        await ctx.defer(ephemeral=True)
        prompt = PromptData(
            value_name="image",
            description="Please attach an image.",
            raw=True)
        _, response = await ctx.prompt(prompt)

        if len(response.attachments) < 1:
            raise commands.BadArgument(
                "Please attach an image you dipshit.")

        async with ctx.typing():
            contents_before = await response.attachments[0].read()
            contents = BytesIO(contents_before)
            classification, confidence = run_on_file(contents)
            contents.seek(0)
            await ctx.respond(content=f"{classification}\n{confidence}", ephemeral=False, file=discord.File(contents, filename="image.png"))


    async def info_error(self, ctx: BlooContext, error):
        if isinstance(error, discord.ApplicationCommandInvokeError):
         error = error.original

        if (isinstance(error, commands.MissingRequiredArgument)
            or isinstance(error, PermissionsFailure)
            or isinstance(error, commands.BadArgument)
            or isinstance(error, commands.BadUnionArgument)
            or isinstance(error, commands.MissingPermissions)
            or isinstance(error, commands.BotMissingPermissions)
            or isinstance(error, commands.MaxConcurrencyReached)
                or isinstance(error, commands.NoPrivateMessage)):
            await ctx.send_error(error)
        else:
            await ctx.send_error("A fatal error occured. Tell <@109705860275539968> about this.")
            logger.error(traceback.format_exc())


def setup(bot):
    bot.add_cog(NeuralNet(bot))