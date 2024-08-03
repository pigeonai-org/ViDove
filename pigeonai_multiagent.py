from openai import OpenAI
from google.colab import userdata


content = "Blizzard Entertainment began planning StarCraft in 1995 with a development team led by Metzen and Phinney. The game debuted at the 1996 Electronic Entertainment Expo and used a modified Warcraft II game engine. StarCraft also marked the creation of Blizzard Entertainment's film department; the game introduced high quality cinematics integral to the storyline of the series. Most of the original development team for StarCraft returned to work on the game's expansion pack, Brood War; that game's development began only shortly after StarCraft was released. In 2001, StarCraft: Ghost began development under Nihilistic Software. Unlike the previous real-time strategy games in the series, Ghost was to be a stealth-action game. After three years of development, work on the game was postponed in 2004. Development of a true RTS sequel, StarCraft II: Wings of Liberty, began in 2003; the game was announced in May 2007 and was released in July 2010. StarCraft II continued with the StarCraft II: Heart of the Swarm expansion, which was released in March 2013. The third and final StarCraft II installment, Legacy of the Void, was released in November 2015. In 2016, a single-player nine-mission pack, Nova Covert Ops, was released in form of DLC."
pos_tag = []
max_iterations = 100
domain = "Starcraft2"
source_language = "English"
target_language = "Chinese"
target_country = "China"
client = OpenAI(
    api_key="INPUT YOUR API KEY HERE"
)


def addition_by_subtraction_collaboration(content, pos_tag, max_iterations, domain, source_language, target_language):
    current_iteration = 0
    history = None

    # Translator Agent
    translation_prompt = f"""This is an {source_language} to {target_language} translation in the field of {domain}, please provide the {target_language} translation for this text.\
    Do not provide any explanations or text apart from the translation. {source_language}: {content} {target_language}:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": translation_prompt
            }
        ]
    )
    history = response.choices[0].message.content

    while current_iteration <= max_iterations:

        # Suggestions Agent

        reflection_prompt = f"""Your task is to carefully read a content in the {history} and a translation from {source_language} to {target_language}, and then give constructive criticism and helpful suggestions to improve the translation. \
        The final style and tone of the translation should match the style of {target_language} colloquially spoken in {target_country}. When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
        (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
        (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
        (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
        (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_language}).
        Write a list of specific, helpful and constructive suggestions for improving the translation.
        Each suggestion should address one specific part of the translation.
        Output only the suggestions and nothing else."""


        response =client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": reflection_prompt
                }
            ]
        )
        suggestion = response.choices[0].message.content

        # Editor Agent
        prompt = f"""Your task is to carefully read, then edit, a translation of the content in the {history} from {source_language} to {target_language}, taking into\
        account a list of expert suggestions and constructive criticisms.

        Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:
        (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
        (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions),
        (iii) style (by ensuring the translations reflect the style of the source text)
        (iv) terminology (inappropriate for context, inconsistent use), or
        (v) other errors.

        Output only the new translation and nothing else."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        reply = response.choices[0].message.content

        if history == reply:
          return reply
        else:
          content = reply
          history = reply
          current_iteration += 1

addition_by_subtraction_collaboration(content, pos_tag, max_iterations, domain, source_language, target_language)

