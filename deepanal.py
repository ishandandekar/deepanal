from argparse import ArgumentParser

from catppuccin.extras.rich_ctp import mocha
from rich.console import Console
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
    cns = Console(theme=mocha)
    cns.print(banner, style="mauve")

    parser = ArgumentParser(
        prog="deepanal", description="Deep analysis of company using AI"
    )
    parser.add_argument(
        "-c",
        "--company",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--industry",
        type=str,
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--location",
        type=str,
    )

    args = parser.parse_args()

    # TODO: Check if API keys are enabled, even check Ollama is installed in path or not


if __name__ == "__main__":
    main()
