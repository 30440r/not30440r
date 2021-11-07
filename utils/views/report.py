import discord
from discord import ui
from discord.ext.commands import Context
from utils.permissions.permissions import permissions

class ReportActions(ui.View):
    def __init__(self, author: discord.Member):
        super().__init__()
        self.author = author

    async def start(self, ctx: Context):
        self.ctx = ctx
        await self.wait()
        
    def check(self, interaction: discord.Interaction):
        if not permissions.has(self.author.guild, interaction.user, 5):
            return False
        return True

    @ui.button(emoji="✅", label="Dismiss", style=discord.ButtonStyle.primary)
    async def dismiss(self, button: ui.Button, interaction: discord.Interaction):
        if not self.check(interaction):
            return
        await self.ctx.message.delete()
        
    @ui.button(emoji="🆔", label="Post ID", style=discord.ButtonStyle.primary)
    async def id(self, button: ui.Button, interaction: discord.Interaction):
        if not self.check(interaction):
            return
        await self.ctx.channel.send(self.author.id)

    @ui.button(emoji="🧹", label="Clean up", style=discord.ButtonStyle.primary)
    async def purge(self, button: ui.Button, interaction: discord.Interaction):
        if not self.check(interaction):
            return
        await self.ctx.channel.purge(limit=100)

