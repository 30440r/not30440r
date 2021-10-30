from discord.commands import Option, slash_command
from discord.ext import commands
from discord.member import Member
from discord.utils import escape_markdown
from data.model.case import Case
from data.services import guild_service, user_service
from utils.checks import mod_and_up, whisper
import utils.checks as checks
from utils.config import cfg
from utils.context import BlooContext
from utils.slash_perms import slash_perms
from utils.mod_logs import prepare_warn_log


"""
Make sure to add the cog to the initial_extensions list
in main.py
"""

class ModActions(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print(checks.Member.__name__)

    @whisper()
    @mod_and_up()
    @slash_command(guild_ids=[cfg.guild_id], description="Warn a user", permissions=slash_perms.mod_and_up())
    async def warn(self, ctx: BlooContext, user: Option(checks.Member, description="User to warn"), points:Option(int, description="Amount of points to warn for"), reason: Option(str, description="Reason for warn", required=False) = "No reason."):
        """Warn a user (mod only)

        Example usage
        --------------
        !warn <@user/ID> <points> <reason (optional)>
        
        Parameters
        ----------
        user : discord.Member
            "The member to warn"
        points : int
            "Number of points to warn far"
        reason : str, optional
            "Reason for warning, by default 'No reason.'"

        """
        if points < 1:  # can't warn for negative/0 points
            raise commands.BadArgument(message="Points can't be lower than 1.")

        guild = guild_service.get_guild()

        reason = escape_markdown(reason)

        # prepare the case object for database
        case = Case(
            _id=guild.case_id,
            _type="WARN",
            mod_id=ctx.author.id,
            mod_tag=str(ctx.author),
            reason=reason,
            punishment=str(points)
        )

        # increment case ID in database for next available case ID
        guild_service.inc_caseid()
        # add new case to DB
        user.add_case(user.id, case)
        # add warnpoints to the user in DB
        user_service.inc_points(user.id, points)

        # fetch latest document about user from DB
        results = user_service.get_user(user.id)
        cur_points = results.warn_points

        # prepare log embed, send to #public-mod-logs, user, channel where invoked
        log = await prepare_warn_log(ctx.author, user, case)
        log.add_field(name="Current points", value=cur_points, inline=True)

        log_kickban = None
        dmed = True
        
        if cur_points >= 600:
            # automatically ban user if more than 600 points
            try:
                await user.send(f"You were banned from {ctx.guild.name} for reaching 600 or more points.", embed=log)
            except Exception:
                dmed = False

            log_kickban = await self.add_ban_case(ctx, user, "600 or more warn points reached.")
            await user.ban(reason="600 or more warn points reached.")

        elif cur_points >= 400 and not results.was_warn_kicked and isinstance(user, Member):
            # kick user if >= 400 points and wasn't previously kicked
            await ctx.settings.set_warn_kicked(user.id)

            try:
                await user.send(f"You were kicked from {ctx.guild.name} for reaching 400 or more points. Please note that you will be banned at 600 points.", embed=log)
            except Exception:
                dmed = False

            log_kickban = await self.add_kick_case(ctx, user, "400 or more warn points reached.")
            await user.kick(reason="400 or more warn points reached.")

        else:
            if isinstance(user, Member):
                try:
                    await user.send(f"You were warned in {ctx.guild.name}. Please note that you will be kicked at 400 points and banned at 600 points.", embed=log)
                except Exception:
                    dmed = False

        # also send response in channel where command was called
        await ctx.respond(embed=log)
        # await ctx.message.delete(delay=10)

        public_chan = ctx.guild.get_channel(
            guild.channel_public)
        if public_chan:
            log.remove_author()
            log.set_thumbnail(url=user.avatar)
            await public_chan.send(user.mention if not dmed else "", embed=log)

            if log_kickban:
                log_kickban.remove_author()
                log_kickban.set_thumbnail(url=user.avatar)
                await public_chan.send(embed=log_kickban)


def setup(bot):
    bot.add_cog(ModActions(bot))