# Source this file from ~/.config/fish/config.fish on the Astrodrift host:
# test -f /software/astrodrift/nix/remote-shell/fish.fish; and source /software/astrodrift/nix/remote-shell/fish.fish

# Interactive SSH sessions enter the development environment and remain in
# Fish. IN_NIX_SHELL prevents the inner Fish process from re-entering Nix.
if status is-interactive; and isatty stdout; and set -q SSH_CONNECTION; and not set -q IN_NIX_SHELL
    if test -f /software/astrodrift/flake.nix; and command -q nix
        cd /software/astrodrift
        nix develop . -c fish
    end
end

# Fish 4 does not expose the command passed to `fish -c` in $argv while this
# file is sourced, so recover it from argv[3] in /proc/self/cmdline.
if not status is-interactive; and set -q SSH_CONNECTION; and not set -q IN_NIX_SHELL
    set -l cmd (string split0 </proc/self/cmdline)[3]
    if set -q cmd[1]; and string match -q -- '*zed*' $cmd
        if command -q nix; and test -f /software/astrodrift/flake.nix; and nix flake metadata /software/astrodrift >/dev/null 2>&1
            exec env ASTRODRIFT_QUIET=1 nix develop /software/astrodrift -c bash -c $cmd
        else
            exec /bin/sh -c $cmd
        end
    end
end
