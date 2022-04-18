from PyInquirer import (Token, ValidationError,
                                    Validator, print_json, prompt, style_from_dict)
from termcolor import colored
from pyfiglet import figlet_format

class CLI:
    version = "1"

    def __init__(self):
        self.style = style_from_dict({
            Token.QuestionMark: '#fac731 bold',
            Token.Answer: '#4688f1 bold',
            Token.Instruction: '',  # default
            Token.Separator: '#cc5454',
            Token.Selected: '#0abf5b',  # default
            Token.Pointer: '#673ab7 bold',
            Token.Question: '',
        })
        self.answers = None
        self.title()

    def title(self, name='ENR', font='standard', color='blue'):
        text = str.strip(figlet_format(name, font=font))
        print(colored(text, color), end='')
        print()

    def askTestInfo(self):
        questions = [
            {'type': 'list',
             'name': 'method',
             'message': 'ENR/NR',
             'default': 'ENR',
             'choices': ['ENR', 'NR']},

            {'type': 'input',
             'name': 'func',
             'message': 'System to test',
             'default': '0'},

            # {'type': 'confirm',
            #  'name': 'span',
            #  'message': 'Run across all c configurations',
            #  'default': False,
            #  'when': lambda answers: answers['method'] == 'ENR'},

            # {'type': 'list',
            #  'name': 'config',
            #  'message': 'Choose c configurations',
            #  'choices': ['2x', 'x+1'],
            #  'when': cConfig},

            {'type': 'input',
             'name': 'range',
             'message': 'Test range',
             'default': '[-50, 50]'},

            {'type': 'input',
             'name': 'resolution',
             'message': 'Resoultion of result',
             'default': '512'},

            {'type': 'confirm',

             'name': 'show',
             'message': 'Show output',
             'default': 'False'
             },

            {'type': 'confirm',
             'name': 'save',
             'message': 'Autosave files',
             'default': False,
             },

        ]
        answers = prompt(questions, style=self.style)
        return answers
