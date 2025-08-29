import os
import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from rich.logging import RichHandler
from catppuccin.extras.rich_ctp import mocha
from rich.console import Console
from rich.traceback import install as setup_tb

from src.graph import workflow

banner = """
██████╗ ███████╗███████╗██████╗  █████╗ ███╗   ██╗ █████╗ ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔══██╗██║
██║  ██║█████╗  █████╗  ██████╔╝███████║██╔██╗ ██║███████║██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══██║██║╚██╗██║██╔══██║██║
██████╔╝███████╗███████╗██║     ██║  ██║██║ ╚████║██║  ██║███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
"""


def main():
    parser = ArgumentParser(
        prog="deepanal", description="Deep analysis of company using AI"
    )
    parser.add_argument("-c", "--company", type=str, required=True)
    parser.add_argument("-i", "--industry", type=str, required=True)
    parser.add_argument("-u", "--url", type=str, required=True)
    parser.add_argument("-l", "--location", type=str, required=True)

    cns = Console(theme=mocha, log_time=True)
    cns.print(banner, style="mauve")
    setup_tb(console=cns)
    args = parser.parse_args()
    user_request = vars(args)

    # TODO: Check if API keys are enabled, even check Ollama is installed in path or not
    if "GOOGLE_API_KEY" not in os.environ:
        msg = "No GOOGLE_API_KEY found. Please setup the environment variable"
        cns.print(msg)
        sys.exit(1)

    if "TAVILY_API_KEY" not in os.environ:
        msg = "No TAVILY_API_KEY found. Please setup the environment variable"
        cns.print(msg)
        sys.exit(1)
    formatter = "%(asctime)s | %(name)s | %(lineno)d | %(message)s"

    logging.basicConfig(
        level="NOTSET",  # Set the desired logging level
        format=formatter,
        datefmt="[%X]",
        handlers=[
            RichHandler(console=cns, show_time=False, show_path=False),
            logging.FileHandler(
                "logs/" + datetime.now().strftime("%H-%M-%S_%d%m%Y") + ".log"
            ),
        ],
    )

    log = logging.getLogger("deepanal")
    log.info("User research request: " + str(user_request))


if __name__ == "__main__":
    main()
