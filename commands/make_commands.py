import argparse

from .shemas import CmdCommand


def make_commands(cmdArguments: list[CmdCommand]):
    parser = argparse.ArgumentParser()

    for command in cmdArguments:
        parser.add_argument(command.commandName, action='store_true',
                            default=False, help=command.commandDescribe)

    args, commands = parser.parse_known_args()

    if len(commands) != 1:
        print('cli works only one command at once')
        return

    for command in cmdArguments:
        if commands[0] in command.commandName:
            command.commandAction()
