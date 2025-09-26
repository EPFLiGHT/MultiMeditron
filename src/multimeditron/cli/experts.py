from multimeditron.cli import EPILOG, main_cli
from multimeditron.experts.config_maker import main as config_maker_main
from multimeditron.experts.train_clip import main as train_clip_main
import click

@main_cli.command(epilog=EPILOG)
@click.argument("config_file", type=click.Path(exists=True))
def train(config_file):
    """Run train_clip.py with the specified YAML configuration file."""
    train_clip_main(config_file)


@main_cli.command(epilog=EPILOG)
def config_maker():
    """Run config_maker.py."""
    config_maker_main()