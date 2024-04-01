from dataclasses import dataclass


@dataclass
class CmdCommand():
    commandName: str
    commandAction: callable
    commandDescribe: str
