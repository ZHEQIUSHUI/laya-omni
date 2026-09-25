"""laya-omni command line: `laya-omni serve ...` starts the web demo."""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: laya-omni serve --fusion RUN --laya LAYA [--image-encoder DIR] [--audio-encoder DIR] [--examples DATA] [--port 8030]")
        return 0
    if argv[0] == "serve":
        from .server import main as serve

        return serve(argv[1:])
    print(f"unknown command: {argv[0]}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
