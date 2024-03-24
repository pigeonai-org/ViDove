import time
from openai import OpenAI
from .abs_model import AbsModel

SUPPORT_DOMAIN = ["SC2"]
ID_MAP = {"SC2": "asst_v1cEwYXexhfmkPEYtISyIjYI"} # should move to secrete place in the future

class Assistant(AbsModel):   
    def __init__(self, system_prompt, temp = 0.15, domain = "SC2"):
        self.client = OpenAI()
        self.thread_id = self.client.beta.threads.create().id
        if domain not in SUPPORT_DOMAIN:
            raise NotImplementedError
        self.assistant_id = ID_MAP[domain]
        self.system_prompt = system_prompt
        self.temp = temp
        
    def send_request(self, input):
        """
        Translates input sentence with desired LLM.

        :param input: Sentence for translation.
        """

        thread_message = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content= self.system_prompt  + "/n" + input
        )

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )

        run = self.wait_on_run(run)

        # retrieve all messages added after our last user message

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id, order="asc", after=thread_message.id
        ).data

        return messages[0].content[0].text.value.strip()

    def wait_on_run(self, run):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id,
            )
            # print(run.status)
            time.sleep(0.5)
        return run
    