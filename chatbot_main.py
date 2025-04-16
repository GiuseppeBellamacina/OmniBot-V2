import asyncio

from dotenv import find_dotenv, load_dotenv

from utilities.colorize import color
from chat.chatbot.session import Session


async def main():
    print(color("[Main]", True, "green"), ": Starting session...", sep="")

    load_dotenv(find_dotenv())
    session = Session(page_title="Chatbot", title="Il tuo Assistente 🤖", icon="🤖")
    session.initialize_session_state()
    await session.update()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    finally:
        loop.run_until_complete(main())
