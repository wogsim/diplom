import pandas as pd
import Typer
from data.collector import VkWallGet, TelegramChanelParser
from data.api_key import VK_ACCESS_TOKEN
from logger import Logger

app = Typer.Typer()

@app.command()
def main() -> None:
    pass