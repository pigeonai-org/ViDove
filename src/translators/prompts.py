system_prompt = (
    "You are a professional translator. your job is to translate texts in domain of {domain} from {source_language} to {target_language} \n"
    "you will be provided with a segment in source language parsed by line, where your translation text should keep the original meaning and the number of lines. \n"
    "Keep every '\n' in the translated text in the corresponding place, and make sure to keep the same number of lines in the translated text. \n"
    "You must break the translated sentence into multiple lines accordingly if original text breaks a complete sentence into different lines\n"
    "You should only output the translated text line by line without any other notation. \n"
)

input_prompt_base = "You current task is to translate the script in the domain of {domain} from {source_language} to {target_language} \n"

input_prompt_supporting_instruction = (
    "Here are some supporting information including previous translation history, context documenting, supporting documents from internet and video clips description that might help you translate the text. \n"
    "Please refer to them if necessary. \n"
)

input_prompt_history = (
    "Previous translation history:\n"
    "{history_str}\n"  # Translation history
)

input_prompt_context = (
    "if you detect any word is in the following context, please use it as a reference for current translation\n"
    "{context_str}"  # Knowledge base
)

input_prompt_supporting_documents = (
    "Here are some supporting documents that might help you translate the text, refer to them if necessary.: "
    "{supporting_documents}"  # Wikipedia, documents from web search
)

input_prompt_video_clips_description = (
    "Here are some descriptions of video clips that might help you translate the text, refer to them if necessary. : "
    "{video_clips_description}"  # Vide clips description
)

input_prompt_query = (
    "Now please translate the following text from {source_language} to {target_language} \n"
    "{query_str}"  # Text to be translated
    "Your translation:"
)


def get_input_prompt(domain, source_language, target_language, input_dict):
    input_parts = []

    input_parts.append(
        input_prompt_base.format(
            domain=domain,
            source_language=source_language,
            target_language=target_language,
        )
    )

    if len(input_dict) > 1:
        input_parts.append(input_prompt_supporting_instruction)

    if "history_str" in input_dict:
        input_parts.append(
            input_prompt_history.format(history_str=input_dict["history_str"])
        )

    if "context_str" in input_dict:
        input_parts.append(
            input_prompt_context.format(context_str=input_dict["context_str"])
        )

    if "supporting_documents" in input_dict:
        input_parts.append(
            input_prompt_supporting_documents.format(
                supporting_documents=input_dict["supporting_documents"]
            )
        )

    if "video_clips_description" in input_dict:
        input_parts.append(
            input_prompt_video_clips_description.format(
                video_clips_description=input_dict["video_clips_description"]
            )
        )

    input_parts.append(
        input_prompt_query.format(
            source_language=source_language,
            target_language=target_language,
            query_str=input_dict["query_str"],
        )
    )

    return "--------- \n ".join(input_parts)
