from factscore.lm import LM
import openai
from openai import AzureOpenAI
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key", azure_endpoint=None, api_version="2023-07-01-preview"):
        self.model_name = model_name
        self.key_path = key_path
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline().strip()

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(self.client, message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            output = response["choices"][0]["message"]["content"]
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(self.client, prompt, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()

def call_ChatGPT(client, message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = client.chat.completions.create(model=model_name,
                                                      messages=message,
                                                      max_tokens=max_len,
                                                      temperature=temp)
            received = True
        except Exception as e:
            num_rate_errors += 1
            if isinstance(e, openai.BadRequestError):
                # something is wrong: e.g. prompt too long
                logging.critical(f"BadRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False

            logging.error("API error: %s (%d). Waiting %dsec" % (e, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(client, prompt, model_name="gpt-3.5-turbo-instruct", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = client.completions.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except Exception as e:
            num_rate_errors += 1
            if isinstance(e, openai.BadRequestError):
                logging.critical(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (e, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
