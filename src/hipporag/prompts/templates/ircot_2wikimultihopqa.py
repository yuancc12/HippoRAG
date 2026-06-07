from .ircot_hotpotqa import ircot_system

prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"}
]
