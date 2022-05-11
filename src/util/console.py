"""

    Used for formatting data that is written to console.

"""


class HexCodes:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def write_normal_message(text):
    print(text)


def write_success_message(text):
    print(f"{HexCodes.OKGREEN}{text}{HexCodes.ENDC}")


def write_fail_message(text):
    print(f"{HexCodes.FAIL}{text}{HexCodes.ENDC}")


def write_warning_message(text):
    print(f"{HexCodes.WARNING}{text}{HexCodes.ENDC}")


def write_header_message(text):
    print(f"{HexCodes.HEADER}{text}{HexCodes.ENDC}")
