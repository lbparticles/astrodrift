# Source this file from ~/.zshenv on the Astrodrift host:
# [[ -f /software/astrodrift/nix/remote-shell/zsh.zsh ]] && source /software/astrodrift/nix/remote-shell/zsh.zsh

# Interactive SSH sessions enter the development environment and remain in
# Zsh. IN_NIX_SHELL prevents the inner Zsh process from re-entering Nix.
if [[ -o interactive && -t 1 && -n "$SSH_CONNECTION" && -z "$IN_NIX_SHELL" ]]; then
    if [[ -f /software/astrodrift/flake.nix ]] && command -v nix >/dev/null 2>&1; then
        cd /software/astrodrift
        nix develop . -c zsh
    fi
fi

# Non-interactive remote commands expose the original `zsh -c` string here.
# Only the Zed startup command is handed to the development environment.
if [[ ! -o interactive && -n "$SSH_CONNECTION" && -z "$IN_NIX_SHELL" \
      && -n "$ZSH_EXECUTION_STRING" && "$ZSH_EXECUTION_STRING" == *zed* ]]; then
    if command -v nix >/dev/null 2>&1 && [[ -f /software/astrodrift/flake.nix ]] \
       && nix flake metadata /software/astrodrift >/dev/null 2>&1; then
        exec env ASTRODRIFT_QUIET=1 nix develop /software/astrodrift -c bash -c "$ZSH_EXECUTION_STRING"
    else
        exec /bin/sh -c "$ZSH_EXECUTION_STRING"
    fi
fi
