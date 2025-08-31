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
    parser.add_argument("-u", "--company_url", type=str, required=True)
    parser.add_argument("-l", "--location", type=str, required=True)

    cns = Console(theme=mocha, log_time=True)
    cns.print(banner, style="mauve")
    setup_tb(console=cns)
    args = parser.parse_args()
    user_request = vars(args)

    if "GOOGLE_API_KEY" not in os.environ:
        msg = "No GOOGLE_API_KEY found. Please setup the environment variable"
        cns.print(msg)
        sys.exit(1)

    if "TAVILY_API_KEY" not in os.environ:
        msg = "No TAVILY_API_KEY found. Please setup the environment variable"
        cns.print(msg)
        sys.exit(1)

    formatter = "%(asctime)s | %(filename)s | %(lineno)d | %(message)s"
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

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    httpx_logger = logging.getLogger("httpcore")
    httpx_logger.setLevel(logging.WARNING)
    httpx_logger = logging.getLogger("asyncio")
    httpx_logger.setLevel(logging.WARNING)

    log = logging.getLogger("deepanal")
    log.info("User research request: " + str(user_request))
    log.info("Initiating DeepAnal research for: " + user_request["company"])
    graph_inputs = {"console": cns, "logger": log}
    graph_inputs.update(user_request)
    log.info("Graph inputs: " + str(graph_inputs))
    workflow.invoke(graph_inputs)


if __name__ == "__main__":
    main()
