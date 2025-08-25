from rich.console import Console
from catppuccin.extras.rich_ctp import mocha


def main():
    cns = Console(theme=mocha)
    cns.print("[bold]hello[/bold]", style="yellow")


if __name__ == "__main__":
    main()
