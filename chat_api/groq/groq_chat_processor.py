import json
from groq import Groq
from typing import List, Dict, Union
from chat_api.models import User
from chat_api.models.conversation import Conversation
from chat_api.models.preference import Preference



class ChatProcessor:
    def __init__(self, user_id: int, conversation: Conversation,  model: str = 'llama-3.1-70b-versatile'):
        self.client = Groq()
        self.model = model
        self.user = User.objects.get(pk=user_id)
        self.conversation = conversation


    def create_message(self, content: str, role: str = 'user') -> Dict[str, Union[str]]:
        return {'role': role, 'content': content}

    def gather_preferences(self, args) -> str:
        data_dict = json.loads(args)
        Preference.objects.create(user=self.user, **data_dict)
        self.conversation.preferences_gathered = True
        self.conversation.save()

        system_message = [
            {
                "role": "system",
                "content": "The preferences from the user have been gathered and saved successfully. Please generate a sign-off message asking if you can help with anything else."
            }
        ]

        # Generate sign-off message
        sign_off_message = self.process_chat(system_message)

        return sign_off_message

    def process_chat(self, history_messages: List[Dict[str, Union[str]]], temperature=0.5, max_tokens=4096, top_p=1, stream=True, stop=None) -> str:
        for message in history_messages:
            if message['content'].upper() == "RESET":
                self.reset_conversation()
                return "The conversation has been reset."

        if not history_messages:
            history_messages.append({
                'role': 'assistant',
                'content': ''
            })

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
                        "required": ["attending", "guest_number",  "needs_accommodation_help", "food_preference"],
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
                    "content": "You are the assistant of Sidzina World Series Open Chess Tournament host."
                               "The tournament is a bit of a joke and an excuse for old Friends to meet. This edition its taking place in Sidzina 735 from 27 till 29 September 2024"
                               "Tournament organizer Mateusz Zięba would like to gather various information's"
                               " (preferences) from his guests."
                               "Please great and asure them you are here to help"
                               "Be helpful"
                               "don't go straight into asking questions but run the conversation in the way that after you gather all necessary information's you"
                               "which is: "
                               "attending, days_attending, guest_number, guest_names only if many guests, needs_accommodation_help, food_preference, interested_in_top_of_babia_gora, "
                               "IF NOT ATTENDING the only information you should collect is the reason why"
                               "REMINDER  keep in mind that is should be natural, ask the questions one by one and try to introduce a bit of small talk "
                               "save the preferences into db."
                               "DON'T make a tool call until you have all the data"
                               "AFTER saving the preferences ask the user if you can help him further"

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
                    if function_name == 'gather_preferences' and self.conversation.preferences_gathered:
                        continue
                    response += function_to_call(function_args)

            response += chunk.choices[0].delta.content or ''

        return response

    def reset_conversation(self):
        self.conversation.messages.all().delete()
        self.conversation.preferences_gathered = False
        self.conversation.save()
