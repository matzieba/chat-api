import json

from groq import Groq
from typing import List, Dict, Union

from chat_api.models import User
from chat_api.models.conversation import Conversation
from chat_api.models.preference import Preference
from chat_api.serializers.preference import PreferenceSerializer


class ChatProcessor:
    def __init__(self, user_id: int,  model: str = 'llama-3.1-70b-versatile'):
        self.client = Groq()
        self.model = model
        self.user = User.objects.get(pk=user_id)


    def create_message(self, content: str, role: str = 'user') -> Dict[str, Union[str]]:
        return {'role': role, 'content': content}

    def gather_preferences(self, args) -> str:
        data_dict = json.loads(args)
        Preference.objects.create(
            user=self.user,
            **data_dict
        )
        return "Your preferences have been saved. Anything else i can help you with?"


    def process_chat(self, history_messages: List[Dict[str, Union[str]]], temperature=1, max_tokens=4096, top_p=1, stream=True, stop=None) -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "gather_preferences",
                    "description": "Collect the users preferences regarding the tournament by asking the questions one by one. <important>Ask all he questions and gather all the data, before trying to save</important> ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "attending": {
                                "type": "boolean",
                                "description": "Is the user attending?"
                            },
                            "days_attending": {
                                "type": "integer",
                                "description": "How many days is the user attending?",
                            },
                            "guest_number": {
                                "type": "integer",
                                "description": "How many guests is the user bringing?"
                            },
                            "guest_names": {
                                "type": "array",
                                "description": "Names of the guests"
                            },
                            "needs_accommodation_help": {
                                "type": "boolean",
                                "description": "Does the user need help with accommodation?"
                            },
                            "food_preference": {
                                "type": "string",
                                "description": "Food preferences of the user"
                            },
                            "interested_in_top_of_babia_gora": {
                                "type": "boolean",
                                "description": "Is the user interested to greet 29.09 at the top of Babia Gora?"
                            },
                            "not_attending_reason": {
                                "type": "string",
                                "description": "The reason for not attending"
                            }
                        },
                        "required": [],
                    },

                }
            }
        ]

        available_functions = {
            "gather_preferences": self.gather_preferences,
        }

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are the assistant of Sidzina World Series Open Chess Tournament."
                               "The tournament is a bit of a joke and an excuse for old Friends to meet. "
                               "Tournament organizer Mateusz Zięba would like to gather various information's"
                               " (preferences) from his "
                               "guests."
                               "Run the conversation in the way that after you gather all necessary informations you "
                               "save the preferences into db.",
                },
                *history_messages,
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice="auto",
            top_p=top_p,
            stream=stream,
            stop=stop,
            tools=tools,
        )

        response = ''

        for chunk in completion:
            response_message = chunk.choices[0].delta
            tool_calls = response_message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    function_to_call = available_functions[function_name]
                    response += function_to_call(function_args)

            response += chunk.choices[0].delta.content or ''

        return response
