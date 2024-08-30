import random

class Rand_prompt:
    def __init__(self, prompt_file):
        self.prompt_file = prompt_file

    def pick(self):
        f = open(self.prompt_file, 'r')

        lines = f.readlines()
        data_len = len(lines)

        rand_num = random.randint(0, data_len+1)

        return lines[rand_num].strip()
