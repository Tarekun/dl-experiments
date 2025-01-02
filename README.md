This repository is an empty template for this Gitea installation. It has a preconfigured project, has actions already enabled, proper desired repo settings that are not enabled by default and they're documented here. This file also includes various tips and tricks here


## Cloning a repo
Avoid cloning repositories via https as the managing of PATs and certificates get messy. Also ssh isn't running on the standard 22 port on the server so to properly clone repos via ssh use the command

`git clone ssh://git@iaisydanieletarek.ddns.net:222/path/to/repo`

## Preconfiguration
The repository contains:
- A project with a base kanban board
- A minimal set of useful labels to be extendend with project specific ones
- Actions enabled by default
- Branch deletion on merge enabled by default
- Branch protection rules on main branch
- `.vscode/settings.json` file with autoformatting for languages frequently used